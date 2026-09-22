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
_TIMEOUT = 15
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
