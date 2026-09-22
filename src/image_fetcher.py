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


def _cache_path(query: str) -> Path:
    key = hashlib.md5(query.lower().encode("utf-8")).hexdigest()[:16]
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


def _search_pexels(query: str) -> str | None:
    key = env("PEXELS_API_KEY")
    if not key:
        return None
    try:
        r = requests.get(
            "https://api.pexels.com/v1/search",
            params={"query": query, "per_page": 5, "orientation": "landscape"},
            headers={"Authorization": key},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        photos = r.json().get("photos", [])
        if photos:
            return photos[0]["src"]["large2x"]
    except Exception as e:  # noqa: BLE001
        log.debug("Pexels lỗi: %s", e)
    return None


def _search_openverse(query: str) -> str | None:
    try:
        r = requests.get(
            "https://api.openverse.org/v1/images/",
            params={
                "q": query,
                "page_size": 5,
                "license_type": "all",
                "aspect_ratio": "wide",
                "mature": "false",
            },
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        for item in results:
            url = item.get("url")
            if url:
                return url
    except Exception as e:  # noqa: BLE001
        log.debug("Openverse lỗi: %s", e)
    return None


def fetch_image(query: str) -> Path | None:
    """Trả về đường dẫn ảnh cho từ khóa, hoặc None nếu không tìm được."""
    query = (query or "").strip()
    if not query:
        return None

    cached = _cache_path(query)
    if cached.exists():
        return cached

    for search in (_search_pexels, _search_openverse):
        url = search(query)
        if url and _download(url, cached):
            log.info("Ảnh minh họa '%s' -> %s", query, cached.name)
            return cached

    log.info("Không tìm được ảnh cho '%s', dùng nền gradient", query)
    return None
