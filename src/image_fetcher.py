"""Tự tìm & tải ảnh minh họa cho scene dựa trên từ khóa.

Nguồn ảnh miễn phí, ưu tiên không cần API key:
  1. Openverse (CC, không cần key) - mặc định
  2. Pexels (cần PEXELS_API_KEY) - nếu có, chất lượng cao hơn

Ảnh tải về được cache theo từ khóa để lần sau khỏi tải lại. Nếu mọi nguồn
thất bại thì trả về None -> visual_engine vẽ nền gradient như cũ.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import requests

from .config import env

log = logging.getLogger(__name__)

_CACHE = Path(__file__).resolve().parent.parent / "assets" / "image_cache"
_VIDEO_CACHE = Path(__file__).resolve().parent.parent / "assets" / "video_cache"
_TIMEOUT = 15
_VIDEO_TIMEOUT = 60
_HEADERS = {"User-Agent": "tech-video-bot/1.0"}


def _cache_path(query: str, index: int = 0) -> Path:
    key = hashlib.md5(f"{query.lower()}|{index}".encode("utf-8")).hexdigest()[:16]
    return _CACHE / f"{key}.jpg"


def _download(url: str, out: Path) -> bool:
    try:
        r = requests.get(url, timeout=_TIMEOUT, headers=_HEADERS, stream=True)
        r.raise_for_status()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        # kiểm tra file ảnh hợp lệ
        if out.stat().st_size < 2000:
            out.unlink(missing_ok=True)
            return False
        return True
    except Exception as e:  # noqa: BLE001
        log.debug("Tải ảnh lỗi %s: %s", url, e)
        out.unlink(missing_ok=True)
        return False


def _search_pexels(query: str, count: int = 8) -> list[str]:
    key = env("PEXELS_API_KEY")
    if not key:
        return []
    try:
        r = requests.get(
            "https://api.pexels.com/v1/search",
            params={"query": query, "per_page": count, "orientation": "landscape"},
            headers={"Authorization": key},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        photos = r.json().get("photos", [])
        return [p["src"]["large2x"] for p in photos if p.get("src")]
    except Exception as e:  # noqa: BLE001
        log.debug("Pexels lỗi: %s", e)
    return []


def _search_openverse(query: str, count: int = 8) -> list[str]:
    try:
        r = requests.get(
            "https://api.openverse.org/v1/images/",
            params={
                "q": query,
                "page_size": count,
                "license_type": "all",
                "aspect_ratio": "wide",
                "mature": "false",
            },
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        return [it["url"] for it in results if it.get("url")]
    except Exception as e:  # noqa: BLE001
        log.debug("Openverse lỗi: %s", e)
    return []


def fetch_image(query: str, index: int = 0) -> Path | None:
    """Trả về ảnh thứ ``index`` cho từ khóa (cho phép nhiều ảnh khác nhau/1 từ khóa).

    index=0 lấy ảnh đầu, index=1 ảnh thứ 2... để scene khác nhau có hình khác nhau.
    """
    query = (query or "").strip()
    if not query:
        return None

    cached = _cache_path(query, index)
    if cached.exists():
        return cached

    for search in (_search_pexels, _search_openverse):
        urls = search(query)
        if not urls:
            continue
        url = urls[index] if index < len(urls) else urls[index % len(urls)]
        if _download(url, cached):
            log.info("Ảnh '%s' #%d -> %s", query, index, cached.name)
            return cached

    log.info("Không tìm được ảnh cho '%s' #%d, dùng nền gradient", query, index)
    return None


# ------------------------------ VIDEO b-roll ------------------------------
# Tải clip footage động minh họa (Pexels Videos). Cần PEXELS_API_KEY.
# Nếu không có key hoặc tải lỗi -> trả None, pipeline dùng ảnh tĩnh như cũ.

def _video_cache_path(query: str, index: int, orientation: str) -> Path:
    key = hashlib.md5(f"vid|{query.lower()}|{index}|{orientation}".encode("utf-8")).hexdigest()[:16]
    return _VIDEO_CACHE / f"{key}.mp4"


def _download_video(url: str, out: Path) -> bool:
    try:
        r = requests.get(url, timeout=_VIDEO_TIMEOUT, headers=_HEADERS, stream=True)
        r.raise_for_status()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                f.write(chunk)
        if out.stat().st_size < 20000:  # file quá nhỏ = hỏng
            out.unlink(missing_ok=True)
            return False
        return True
    except Exception as e:  # noqa: BLE001
        log.debug("Tải video lỗi %s: %s", url, e)
        out.unlink(missing_ok=True)
        return False


def _search_pexels_videos(query: str, orientation: str, count: int = 10) -> list[str]:
    """Trả về danh sách link file .mp4 từ Pexels Videos (ưu tiên độ phân giải HD gần 1080)."""
    key = env("PEXELS_API_KEY")
    if not key:
        return []
    try:
        r = requests.get(
            "https://api.pexels.com/videos/search",
            params={"query": query, "per_page": count, "orientation": orientation},
            headers={"Authorization": key},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        out: list[str] = []
        for v in r.json().get("videos", []):
            files = v.get("video_files", [])
            if not files:
                continue
            # chọn file .mp4 có chiều cao gần 1080 nhất (không quá lớn để tải nhanh)
            mp4s = [f for f in files if f.get("file_type") == "video/mp4" and f.get("link")]
            if not mp4s:
                continue
            best = min(mp4s, key=lambda f: abs((f.get("height") or 0) - 1080))
            out.append(best["link"])
        return out
    except Exception as e:  # noqa: BLE001
        log.debug("Pexels Videos lỗi: %s", e)
    return []


def fetch_video(query: str, index: int = 0, vertical: bool = False) -> Path | None:
    """Tải clip b-roll cho từ khóa. ``vertical`` True cho short 9:16.

    Trả về đường dẫn .mp4 đã cache, hoặc None nếu không có nguồn/không tải được.
    """
    query = (query or "").strip()
    if not query:
        return None
    orientation = "portrait" if vertical else "landscape"
    cached = _video_cache_path(query, index, orientation)
    if cached.exists():
        return cached

    urls = _search_pexels_videos(query, orientation)
    if not urls:
        return None
    url = urls[index] if index < len(urls) else urls[index % len(urls)]
    if _download_video(url, cached):
        log.info("Video b-roll '%s' #%d (%s) -> %s", query, index, orientation, cached.name)
        return cached
    return None
