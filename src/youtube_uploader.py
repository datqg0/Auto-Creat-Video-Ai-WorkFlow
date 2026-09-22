"""Upload video lên YouTube qua Data API v3.

Xác thực OAuth2. Trên máy cá nhân: chạy `python -m src.youtube_uploader --auth`
để tạo token.json. Trên CI: nạp token từ env YOUTUBE_TOKEN + YOUTUBE_CLIENT_SECRET.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from .config import CREDENTIALS_DIR, env

log = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
CLIENT_SECRET_FILE = CREDENTIALS_DIR / "client_secret.json"
TOKEN_FILE = CREDENTIALS_DIR / "token.json"


def _load_credentials() -> Credentials:
    CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)

    # Ưu tiên token từ env (CI)
    token_env = env("YOUTUBE_TOKEN")
    if token_env and not TOKEN_FILE.exists():
        TOKEN_FILE.write_text(token_env, encoding="utf-8")
    secret_env = env("YOUTUBE_CLIENT_SECRET")
    if secret_env and not CLIENT_SECRET_FILE.exists():
        CLIENT_SECRET_FILE.write_text(secret_env, encoding="utf-8")

    creds: Credentials | None = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if creds and creds.valid:
        return creds
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
        return creds

    # Chưa có token hợp lệ -> chạy OAuth flow (chỉ dùng ở local)
    if not CLIENT_SECRET_FILE.exists():
        raise RuntimeError(
            "Thiếu credentials/client_secret.json. Tải từ Google Cloud Console."
        )
    flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET_FILE), SCOPES)
    creds = flow.run_local_server(port=0)
    TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
    return creds


def upload(video_path: Path, meta: dict, thumbnail: Path | None = None) -> str:
    """Upload video, đặt thumbnail. Trả về YouTube video id."""
    creds = _load_credentials()
    youtube = build("youtube", "v3", credentials=creds)

    body = {
        "snippet": {
            "title": meta["title"],
            "description": meta["description"],
            "tags": meta.get("tags", []),
            "categoryId": meta.get("categoryId", "28"),
        },
        "status": {
            "privacyStatus": meta.get("privacyStatus", "public"),
            "selfDeclaredMadeForKids": meta.get("madeForKids", False),
        },
    }

    media = MediaFileUpload(str(video_path), chunksize=-1, resumable=True, mimetype="video/mp4")
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            log.info("Upload %.0f%%", status.progress() * 100)

    video_id = response["id"]
    log.info("Đã upload: https://youtu.be/%s", video_id)

    if thumbnail and thumbnail.exists():
        try:
            youtube.thumbnails().set(
                videoId=video_id, media_body=MediaFileUpload(str(thumbnail))
            ).execute()
        except Exception as e:  # noqa: BLE001 - thumbnail không bắt buộc
            log.warning("Đặt thumbnail lỗi: %s", e)

    return video_id


def _auth_only() -> None:
    """Tạo token.json ở local để copy vào GitHub Secret YOUTUBE_TOKEN."""
    creds = _load_credentials()
    print("Xác thực thành công. token.json đã lưu tại:", TOKEN_FILE)
    print("\nNội dung token (copy vào GitHub Secret YOUTUBE_TOKEN):\n")
    print(json.dumps(json.loads(creds.to_json())))


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--auth", action="store_true", help="Chạy OAuth flow tạo token")
    args = parser.parse_args()
    if args.auth:
        _auth_only()
