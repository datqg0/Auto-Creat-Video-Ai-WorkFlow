"""Upload video lên YouTube qua Data API v3.

Xác thực OAuth2. Trên máy cá nhân: chạy `python -m src.youtube_uploader --auth`
để tạo token.json. Trên CI: nạp token từ env YOUTUBE_TOKEN + YOUTUBE_CLIENT_SECRET.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
import re
import time

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError, ResumableUploadError
from googleapiclient.http import MediaFileUpload

from .config import CREDENTIALS_DIR, env, get_runtime_policy
from .metadata import sanitize_description, sanitize_tags, sanitize_title

log = logging.getLogger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.force-ssl",
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/yt-analytics.readonly",
]
CLIENT_SECRET_FILE = CREDENTIALS_DIR / "client_secret.json"
TOKEN_FILE = CREDENTIALS_DIR / "token.json"


def _load_credentials(allow_interactive: bool = True) -> Credentials:
    CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)

    # Ưu tiên token từ env (CI)
    token_env = env("YOUTUBE_TOKEN")
    if token_env:
        TOKEN_FILE.write_text(token_env, encoding="utf-8")
    secret_env = env("YOUTUBE_CLIENT_SECRET")
    if secret_env:
        CLIENT_SECRET_FILE.write_text(secret_env, encoding="utf-8")

    creds: Credentials | None = None
    if TOKEN_FILE.exists():
        try:
            # Không ép cứng SCOPES ở đây để tránh lỗi invalid_scope nếu token cũ chưa cấp đủ scope mới
            creds = Credentials.from_authorized_user_file(str(TOKEN_FILE))
        except Exception as e:
            log.warning("Không thể đọc token.json: %s", e)
            creds = None

    if creds and creds.valid:
        return creds
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
            return creds
        except Exception as e:
            err_msg = str(e)
            if "invalid_scope" in err_msg:
                log.warning(
                    "Refresh token gặp lỗi invalid_scope (%s). Thử fallback chỉ với scope cơ bản 'youtube.upload'...",
                    e,
                )
                try:
                    creds._scopes = ["https://www.googleapis.com/auth/youtube.upload"]
                    creds.refresh(Request())
                    TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
                    log.info("✓ Fallback refresh với 'youtube.upload' thành công!")
                    return creds
                except Exception as e2:
                    log.error("Fallback refresh vẫn thất bại: %s", e2)
            raise

    # Chế độ nền không được mở trình duyệt nếu token thiếu/hết hạn không thể refresh.
    if not allow_interactive:
        raise RuntimeError("Không có OAuth token hợp lệ; chế độ nền không mở trình duyệt để xác thực.")

    # Chưa có token hợp lệ -> chạy OAuth flow (chỉ dùng ở local)
    if not CLIENT_SECRET_FILE.exists():
        raise RuntimeError(
            "Thiếu credentials/client_secret.json. Tải từ Google Cloud Console."
        )
    flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET_FILE), SCOPES)
    creds = flow.run_local_server(port=0)
    TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
    return creds


def upload_caption(youtube, video_id: str, srt_path: Path, language: str = "vi", name: str = "Tiếng Việt") -> None:
    """Tải file phụ đề .srt lên YouTube Captions."""
    if not srt_path.exists():
        return
    try:
        body = {
            "snippet": {
                "videoId": video_id,
                "language": language,
                "name": name,
                "isDraft": False,
            }
        }
        media = MediaFileUpload(str(srt_path), mimetype="*/*", resumable=True)
        youtube.captions().insert(part="snippet", body=body, media_body=media).execute()
        log.info("Đã upload phụ đề %s (%s) cho video %s", name, language, video_id)
    except Exception as e:  # noqa: BLE001
        if "insufficientPermissions" in str(e):
            log.warning(
                "Upload phụ đề %s lỗi: Token thiếu quyền 'youtube.force-ssl'. Hãy chạy `python -m src.youtube_uploader --auth` để cấp lại token và cập nhật Secret YOUTUBE_TOKEN trên GitHub.",
                language,
            )
        else:
            log.warning("Upload phụ đề %s lỗi: %s", language, e)


def upload(
    video_path: Path,
    meta: dict,
    thumbnail: Path | None = None,
    captions: dict[str, Path] | None = None,
) -> str:
    """Upload video, đặt thumbnail, và upload file phụ đề nếu có. Trả về YouTube video id."""
    creds = _load_credentials()
    youtube = build("youtube", "v3", credentials=creds)

    safe_title = sanitize_title(str(meta.get("title") or "Video"))
    safe_desc = sanitize_description(str(meta.get("description") or ""))
    safe_tags = sanitize_tags(meta.get("tags") or [])

    publish_policy = get_runtime_policy().publishing
    staging_privacy = publish_policy.staging_privacy if publish_policy.mode == "automatic" else meta.get("privacyStatus", "private")
    body = {
        "snippet": {
            "title": safe_title,
            "description": safe_desc,
            "tags": safe_tags,
            "categoryId": str(meta.get("categoryId", "28")),
            "defaultLanguage": meta.get("language", "vi"),
            "defaultAudioLanguage": meta.get("language", "vi"),
        },
        "status": {
            "privacyStatus": staging_privacy,
            "selfDeclaredMadeForKids": meta.get("madeForKids", False),
        },
    }

    def _execute_upload(upload_body: dict) -> dict:
        media = MediaFileUpload(str(video_path), chunksize=-1, resumable=True, mimetype="video/mp4")
        request = youtube.videos().insert(part="snippet,status", body=upload_body, media_body=media)
        res = None
        while res is None:
            status, res = request.next_chunk()
            if status:
                log.info("Upload %.0f%%", status.progress() * 100)
        return res

    try:
        response = _execute_upload(body)
    except (ResumableUploadError, HttpError) as e:
        err_msg = str(e)
        if "invalidDescription" in err_msg or "description" in err_msg.lower():
            log.warning("YouTube từ chối description (%s). Thử lại với description tối giản...", e)
            body["snippet"]["description"] = f"{body['snippet']['title']}\n\n#Tech #Education"
            response = _execute_upload(body)
        elif "invalidTitle" in err_msg or "title" in err_msg.lower():
            log.warning("YouTube từ chối title (%s). Thử lại với title làm sạch...", e)
            body["snippet"]["title"] = re.sub(r"[^\w\s-]", "", body["snippet"]["title"])[:60].strip() or "Tech Visualized"
            response = _execute_upload(body)
        elif "invalidTag" in err_msg or "tag" in err_msg.lower():
            log.warning("YouTube từ chối tags (%s). Thử lại không dùng tags...", e)
            body["snippet"]["tags"] = []
            response = _execute_upload(body)
        else:
            raise

    video_id = response["id"]
    log.info("Đã upload: https://youtu.be/%s", video_id)

    if thumbnail and thumbnail.exists():
        try:
            youtube.thumbnails().set(
                videoId=video_id, media_body=MediaFileUpload(str(thumbnail))
            ).execute()
        except Exception as e:  # noqa: BLE001 - thumbnail không bắt buộc
            log.warning("Đặt thumbnail lỗi: %s", e)

    if captions:
        token_scopes = getattr(creds, "scopes", None) or []
        has_caption_scope = any("youtube.force-ssl" in s or s.endswith("/auth/youtube") for s in token_scopes)
        if not has_caption_scope:
            log.warning(
                "Token hiện tại chỉ có quyền upload video, chưa có quyền 'youtube.force-ssl' để tải phụ đề .srt lên YouTube Captions. "
                "Bỏ qua bước nạp phụ đề. (Để bật phụ đề tự động trên YouTube, hãy chạy local: `python -m src.youtube_uploader --auth` "
                "rồi cập nhật lại Secret YOUTUBE_TOKEN trên GitHub)."
            )
        else:
            for lang_code, cap_path in captions.items():
                if cap_path and cap_path.exists():
                    name = "Tiếng Việt" if lang_code == "vi" else "English"
                    upload_caption(youtube, video_id, cap_path, language=lang_code, name=name)

    return video_id


def get_processing_status(video_id: str) -> dict:
    """Đọc processing và privacy status của một video đã upload."""
    creds = _load_credentials(allow_interactive=False)
    youtube = build("youtube", "v3", credentials=creds)
    response = youtube.videos().list(part="status,processingDetails", id=video_id).execute()
    items = response.get("items") or []
    if not items:
        return {"processing_status": "missing", "privacy_status": None, "failure_reason": "video_not_found"}
    item = items[0]
    processing = item.get("processingDetails") or {}
    status = item.get("status") or {}
    return {
        "processing_status": processing.get("processingStatus", "unknown"),
        "privacy_status": status.get("privacyStatus"),
        "failure_reason": processing.get("processingFailureReason"),
        "rejection_reason": status.get("rejectionReason"),
        "upload_status": status.get("uploadStatus"),
    }


def wait_for_processing(video_id: str) -> dict:
    """Đợi YouTube xử lý xong; timeout thì giữ nguyên private."""
    policy = get_runtime_policy().publishing
    deadline = time.monotonic() + policy.processing_timeout_seconds
    latest: dict = {}
    while time.monotonic() < deadline:
        latest = get_processing_status(video_id)
        state = latest.get("processing_status")
        if state in {"succeeded", "failed", "terminated", "missing"}:
            return latest
        time.sleep(policy.poll_interval_seconds)
    return {**latest, "processing_status": "timeout"}


def set_privacy_status(video_id: str, privacy_status: str) -> None:
    """Chuyển visibility sau khi machine approval và processing đều pass."""
    if privacy_status not in {"public", "private", "unlisted"}:
        raise ValueError(f"privacy status không hợp lệ: {privacy_status}")
    creds = _load_credentials(allow_interactive=False)
    youtube = build("youtube", "v3", credentials=creds)
    current = youtube.videos().list(part="status", id=video_id).execute()
    items = current.get("items") or []
    if not items:
        raise RuntimeError(f"Không tìm thấy YouTube video {video_id} để đổi visibility.")
    status_body = dict(items[0].get("status") or {})
    # Chỉ gửi các field mutable được API chấp nhận; giữ nguyên policy hiện tại.
    allowed = {
        "license", "embeddable", "publicStatsViewable", "selfDeclaredMadeForKids",
        "containsSyntheticMedia",
    }
    status_body = {key: value for key, value in status_body.items() if key in allowed}
    status_body["privacyStatus"] = privacy_status
    youtube.videos().update(
        part="status",
        body={"id": video_id, "status": status_body},
    ).execute()
    log.info("Video %s đã chuyển sang %s", video_id, privacy_status)


def auto_publish_if_ready(
    video_id: str,
    manifest: dict,
    quality_report: dict,
    content_report: dict,
) -> dict:
    """Machine approve, đợi processing và tự public; mọi lỗi đều giữ private."""
    from .publish_policy import evaluate_publish_policy

    decision = evaluate_publish_policy(manifest, quality_report, content_report)
    result = {"decision": decision.as_dict(), "video_id": video_id, "published": False, "released": False}
    if not decision.approved:
        return result
    try:
        processing = wait_for_processing(video_id)
    except Exception as exc:  # noqa: BLE001 - lỗi API phải giữ video private, không gây re-upload
        result["decision"] = {
            "approved": False,
            "action": "quarantine",
            "reasons": [f"Không đọc được YouTube processing status: {exc}"],
        }
        return result
    result["processing"] = processing
    if processing.get("processing_status") != "succeeded":
        result["decision"] = {
            "approved": False,
            "action": "quarantine",
            "reasons": [f"YouTube processing={processing.get('processing_status')}"],
        }
        return result
    final_privacy = get_runtime_policy().publishing.final_privacy
    try:
        set_privacy_status(video_id, final_privacy)
    except Exception as exc:  # noqa: BLE001 - tuyệt đối không làm pipeline upload lại video
        result["decision"] = {
            "approved": False,
            "action": "quarantine",
            "reasons": [f"Không đổi được visibility: {exc}"],
        }
        return result
    result["published"] = final_privacy == "public"
    result["released"] = final_privacy in {"public", "unlisted"}
    result["final_privacy"] = final_privacy
    return result


def _auth_only() -> None:
    """Tạo token.json ở local để copy vào GitHub Secret YOUTUBE_TOKEN."""
    if not CLIENT_SECRET_FILE.exists():
        raise RuntimeError(
            "Thiếu credentials/client_secret.json. Tải từ Google Cloud Console."
        )
    flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET_FILE), SCOPES)
    creds = flow.run_local_server(port=0)
    TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
    print("Xác thực thành công. token.json đã lưu tại:", TOKEN_FILE)
    print("\nNội dung token (copy vào GitHub Secret YOUTUBE_TOKEN):\n")
    print(json.dumps(json.loads(creds.to_json())))


if __name__ == "__main__":
    import argparse
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--auth", action="store_true", help="Chạy OAuth flow tạo token")
    args = parser.parse_args()
    if args.auth:
        _auth_only()
