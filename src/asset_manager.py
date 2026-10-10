"""Kho asset icon và visual elements cho video (tech brands & diagram symbols).

Tự động tải SVG từ SimpleIcons CDN / Iconify và cache cục bộ tại assets/icons/.
Hỗ trợ convert SVG sang PNG nền trong suốt để ghép vào Scene Hook / Đặt vấn đề / Bài tập.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import urllib.request
from pathlib import Path

from PIL import Image

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
_ICONS_DIR = _ROOT / "assets" / "icons"
from .motion_graphics import get_browser_path

# Mapping một số từ khóa phổ biến sang slug chính xác của SimpleIcons
_SLUG_MAP = {
    "c++": "cplusplus",
    "c#": "csharp",
    "node": "nodedotjs",
    "nodejs": "nodedotjs",
    "node.js": "nodedotjs",
    "nextjs": "nextdotjs",
    "next.js": "nextdotjs",
    "vue": "vuedotjs",
    "vuejs": "vuedotjs",
    "aws": "amazonaws",
    "kafka": "apachekafka",
    "spark": "apachespark",
    "tf": "tensorflow",
    "k8s": "kubernetes",
    "postgres": "postgresql",
    "mongo": "mongodb",
    "ai": "openai",
    "llm": "openai",
    "gpt": "openai",
    "gemini": "googlegemini",
    "claude": "anthropic",
}


def _clean_slug(keyword: str) -> str:
    kw = keyword.lower().strip()
    if kw in _SLUG_MAP:
        return _SLUG_MAP[kw]
    # Bỏ ký tự đặc biệt
    kw = re.sub(r"[^a-z0-9]", "", kw)
    return kw


def get_tech_icon(keyword: str, color_hex: str = "38bdf8") -> Path | None:
    """Tìm icon SVG trong cache cục bộ hoặc tải từ SimpleIcons/Iconify."""
    slug = _clean_slug(keyword)
    if not slug:
        return None

    _ICONS_DIR.mkdir(parents=True, exist_ok=True)
    local_svg = _ICONS_DIR / f"{slug}.svg"
    if local_svg.exists() and local_svg.stat().st_size > 100:
        return local_svg

    # 1. Thử tải từ SimpleIcons
    clean_color = color_hex.lstrip("#")
    urls = [
        f"https://cdn.simpleicons.org/{slug}/{clean_color}",
        f"https://api.iconify.design/lucide:{slug}.svg?color=%23{clean_color}",
    ]

    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req, timeout=5) as res:
                if res.status == 200:
                    data = res.read()
                    if b"<svg" in data:
                        local_svg.write_bytes(data)
                        log.info("Đã tải asset icon mới: %s -> %s", slug, local_svg.name)
                        return local_svg
        except Exception:
            continue

    return None


def icon_to_png(svg_path: Path, out_png: Path, size: int = 160) -> Path | None:
    """Render file SVG thành PNG có nền trong suốt (RGBA) ở kích thước chỉ định."""
    if not svg_path.exists():
        return None

    if out_png.exists() and out_png.stat().st_size > 500:
        return out_png

    out_png.parent.mkdir(parents=True, exist_ok=True)

    if os.path.exists(_EDGE_PATH):
        try:
            # Tạo file HTML bọc SVG
            svg_content = svg_path.read_text(encoding="utf-8")
            html = f"""<!DOCTYPE html>
<html>
<body style="margin:0;padding:0;background:transparent;display:flex;align-items:center;justify-content:center;width:{size}px;height:{size}px;overflow:hidden;">
  <div style="width:{int(size*0.85)}px;height:{int(size*0.85)}px;display:flex;align-items:center;justify-content:center;">
    {svg_content}
  </div>
</body>
</html>"""
            tmp_html = out_png.parent / f"_tmp_{svg_path.stem}.html"
            tmp_html.write_text(html, encoding="utf-8")

            cmd = [
                _EDGE_PATH,
                "--headless",
                "--disable-gpu",
                f"--screenshot={out_png.resolve()}",
                f"--window-size={size},{size}",
                "--hide-scrollbars",
                "--default-background-color=00000000",
                str(tmp_html.resolve()),
            ]
            subprocess.run(cmd, timeout=15, capture_output=True)
            tmp_html.unlink(missing_ok=True)
            if out_png.exists() and out_png.stat().st_size > 200:
                return out_png
        except Exception as e:
            log.warning("Chuyển SVG sang PNG lỗi: %s", e)

    return None
