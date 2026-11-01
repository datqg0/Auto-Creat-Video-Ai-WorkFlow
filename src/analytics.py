"""Optional YouTube audience-retention collection and prompt feedback."""
from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .config import CONFIG, OUTPUT_DIR, env

log = logging.getLogger(__name__)
_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{6,}$")
_READ_SCOPE = "https://www.googleapis.com/auth/youtube.readonly"


def _retention_dir() -> Path:
    configured = Path(str((CONFIG.get("analytics", {}) or {}).get("retention_dir", "output/retention")))
    return configured if configured.is_absolute() else OUTPUT_DIR.parent / configured


def save_video_metadata(metadata: dict[str, Any]) -> Path:
    """Persist non-secret upload metadata beside retention reports (CI cache friendly)."""
    video_id = str(metadata.get("youtube_id") or "").strip()
    if not _VIDEO_ID.fullmatch(video_id):
        raise ValueError("YouTube video ID không hợp lệ.")
    folder = _retention_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{video_id}.meta.json"
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _read_metadata(video_id: str) -> dict | None:
    path = _retention_dir() / f"{video_id}.meta.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            log.warning("Metadata retention không đọc được (%s): %s", video_id, exc)
    for manifest_path in OUTPUT_DIR.glob("video_*/manifest.json"):
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            if data.get("youtube_id") == video_id:
                return data
        except Exception:  # noqa: BLE001
            continue
    return None


def _analytics_service():
    from .youtube_uploader import CLIENT_SECRET_FILE, TOKEN_FILE, _load_credentials
    from googleapiclient.discovery import build

    if not env("YOUTUBE_TOKEN") and not TOKEN_FILE.exists():
        raise RuntimeError("Chưa có YOUTUBE_TOKEN; bỏ qua thu thập retention tự động.")
    if not env("YOUTUBE_CLIENT_SECRET") and not CLIENT_SECRET_FILE.exists():
        raise RuntimeError("Thiếu YOUTUBE_CLIENT_SECRET để tải token OAuth.")
    creds = _load_credentials(allow_interactive=False)
    if not creds.has_scopes([_READ_SCOPE]):
        raise PermissionError(
            "OAuth token chưa có youtube.readonly. Chạy `python -m src.youtube_uploader --auth` "
            "và cập nhật lại GitHub Secret YOUTUBE_TOKEN."
        )
    return build("youtubeAnalytics", "v2", credentials=creds, cache_discovery=False)


def fetch_retention(video_id: str, start_date: str, end_date: str, service=None) -> dict:
    if not _VIDEO_ID.fullmatch(video_id):
        raise ValueError("YouTube video ID không hợp lệ.")
    service = service or _analytics_service()
    base = {
        "ids": "channel==MINE",
        "startDate": start_date,
        "endDate": end_date,
        "filters": f"video=={video_id}",
    }
    retention = service.reports().query(
        **base,
        metrics="audienceWatchRatio,relativeRetentionPerformance",
        dimensions="elapsedVideoTimeRatio",
        sort="elapsedVideoTimeRatio",
        maxResults=200,
    ).execute()
    views_result = service.reports().query(**base, metrics="views").execute()
    headers = [column.get("name") for column in retention.get("columnHeaders", [])]
    points = []
    for row in retention.get("rows", []) or []:
        item = dict(zip(headers, row))
        try:
            points.append({
                "elapsed_ratio": float(item["elapsedVideoTimeRatio"]),
                "audience_watch_ratio": float(item["audienceWatchRatio"]),
                "relative_retention": float(item["relativeRetentionPerformance"]),
            })
        except (KeyError, TypeError, ValueError):
            continue
    view_rows = views_result.get("rows") or []
    views = int(view_rows[0][0]) if view_rows and view_rows[0] else 0
    return {"views": views, "points": points, "start_date": start_date, "end_date": end_date}


def _report_path(video_id: str) -> Path:
    return _retention_dir() / f"{video_id}.json"


def collect_recent_retention_feedback() -> int:
    """Fetch ready reports for recently published videos; never block video production."""
    cfg = CONFIG.get("analytics", {}) or {}
    if not cfg.get("enabled", False):
        return 0
    try:
        from . import db

        rows = db.recent_uploaded_videos_for_analytics(int(cfg.get("recent_videos_to_check", 12)))
    except Exception as exc:  # noqa: BLE001
        log.info("Retention: không đọc được lịch sử upload: %s", exc)
        return 0
    if not rows:
        return 0

    now = datetime.now(timezone.utc)
    minimum_age = timedelta(days=int(cfg.get("wait_after_publish_days", 3)))
    refresh_after = timedelta(days=int(cfg.get("refresh_after_days", 7)))
    eligible: list[tuple[dict, dict]] = []
    for row in rows:
        video_id = str(row.get("youtube_id") or "")
        meta = _read_metadata(video_id)
        if not meta:
            continue
        try:
            uploaded_at = datetime.fromisoformat(str(meta.get("uploaded_at") or row.get("updated_at")))
            if uploaded_at.tzinfo is None:
                uploaded_at = uploaded_at.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            uploaded_at = now - minimum_age
        if now - uploaded_at < minimum_age:
            continue
        report_path = _report_path(video_id)
        if report_path.exists():
            try:
                existing = json.loads(report_path.read_text(encoding="utf-8"))
                checked = datetime.fromisoformat(str(existing.get("fetched_at", "")))
                if checked.tzinfo is None:
                    checked = checked.replace(tzinfo=timezone.utc)
                if now - checked < refresh_after:
                    continue
            except Exception:  # noqa: BLE001
                pass
        eligible.append((row, meta))
    if not eligible:
        return 0

    try:
        service = _analytics_service()
    except Exception as exc:  # noqa: BLE001 - OAuth/API missing must not stop rendering
        log.info("Retention chưa được kết nối; pipeline tiếp tục bình thường: %s", exc)
        return 0

    folder = _retention_dir()
    folder.mkdir(parents=True, exist_ok=True)
    saved = 0
    end = now.date() - timedelta(days=3)
    start = end - timedelta(days=int(cfg.get("lookback_days", 28)))
    for row, meta in eligible:
        video_id = str(row.get("youtube_id") or "")
        try:
            data = fetch_retention(video_id, start.isoformat(), end.isoformat(), service)
            report = {
                "youtube_id": video_id,
                "topic": meta.get("topic", row.get("topic", "")),
                "title": meta.get("title", row.get("title", "")),
                "mode": meta.get("mode", "long"),
                "duration_seconds": float(meta.get("duration_seconds") or 0),
                "views": data["views"],
                "points": data["points"],
                "start_date": data["start_date"],
                "end_date": data["end_date"],
                "fetched_at": now.isoformat(),
                "guidance_eligible": data["views"] >= int(cfg.get("min_views_for_guidance", 100)),
            }
            _report_path(video_id).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            saved += 1
            log.info("Retention đã lưu cho %s (%d views, %d điểm).", video_id, data["views"], len(data["points"]))
        except Exception as exc:  # noqa: BLE001
            log.info("Không lấy được retention cho %s; sẽ thử lại lần sau: %s", video_id, exc)
    return saved


def retention_guidance_for_prompt(mode: str) -> str:
    """Summarize only sufficiently sampled reports of the same video format."""
    cfg = CONFIG.get("analytics", {}) or {}
    if not cfg.get("enabled", False):
        return ""
    folder = _retention_dir()
    if not folder.exists():
        return ""
    minimum_views = int(cfg.get("min_views_for_guidance", 100))
    reports = []
    for path in folder.glob("*.json"):
        if path.name.endswith(".meta.json"):
            continue
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
            if item.get("guidance_eligible") and item.get("mode") == mode and item.get("points"):
                reports.append(item)
        except Exception:  # noqa: BLE001
            continue
    reports.sort(key=lambda item: item.get("fetched_at", ""), reverse=True)
    if not reports:
        return ""

    lines = [
        "PHẢN HỒI RETENTION TỪ VIDEO CÙNG ĐỊNH DẠNG (tham khảo xu hướng, không suy diễn nhân quả):"
    ]
    for report in reports[:3]:
        points = sorted(report["points"], key=lambda point: point.get("elapsed_ratio", 0))
        duration = float(report.get("duration_seconds") or 0)
        views = int(report.get("views", 0))
        observations = []
        if duration > 30 and points:
            target = 30.0 / duration
            intro = min(points, key=lambda point: abs(point["elapsed_ratio"] - target))
            if intro.get("audience_watch_ratio") is not None:
                observations.append(f"còn xem ở 30 giây khoảng {intro['audience_watch_ratio']:.0%}")
        drops = []
        for previous, current in zip(points, points[1:]):
            delta = current.get("audience_watch_ratio", 0) - previous.get("audience_watch_ratio", 0)
            if delta <= -0.12:
                second = current["elapsed_ratio"] * duration
                drops.append((delta, second))
        for delta, second in sorted(drops)[:2]:
            observations.append(f"tụt khoảng {abs(delta):.0%} gần {int(second // 60)}:{int(second % 60):02d}")
        if observations:
            lines.append(f"- {report.get('title') or report.get('topic')}: {', '.join(observations)} ({views} lượt xem).")
    if len(lines) == 1:
        return ""
    lines.append(f"Chỉ dùng các quan sát này để đề xuất thử nghiệm hook/nhịp kể; cần tối thiểu {minimum_views} views.")
    return "\n".join(lines)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Lấy báo cáo retention của một video YouTube.")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD; mặc định 28 ngày gần nhất đã xử lý")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD; mặc định lùi 3 ngày")
    args = parser.parse_args()
    end = date.fromisoformat(args.end_date) if args.end_date else datetime.now(timezone.utc).date() - timedelta(days=3)
    start = date.fromisoformat(args.start_date) if args.start_date else end - timedelta(days=int((CONFIG.get("analytics", {}) or {}).get("lookback_days", 28)))
    data = fetch_retention(args.video_id, start.isoformat(), end.isoformat())
    meta = _read_metadata(args.video_id) or {"youtube_id": args.video_id}
    report = {
        **meta,
        **data,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "guidance_eligible": data["views"] >= int((CONFIG.get("analytics", {}) or {}).get("min_views_for_guidance", 100)),
    }
    path = _report_path(args.video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Retention đã lưu: {path} ({data['views']} views, {len(data['points'])} điểm).")


if __name__ == "__main__":
    _cli()
