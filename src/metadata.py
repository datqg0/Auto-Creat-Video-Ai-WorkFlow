"""Sinh metadata YouTube (title/description/tags) và thumbnail 1280x720."""
from __future__ import annotations

import logging
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .config import CONFIG
from .models import Script

log = logging.getLogger(__name__)


def build_metadata(script: Script) -> dict:
    yt = CONFIG["youtube"]
    tags = list(dict.fromkeys([*script.tags, *yt.get("default_tags", [])]))[:15]

    description = script.description.strip()
    description += "\n\n" + "\n".join(f"#{t.replace(' ', '')}" for t in tags[:5])
    description += "\n\nVideo được tạo tự động bằng visualization engine."

    return {
        "title": script.title[:100],
        "description": description[:4900],
        "tags": tags,
        "categoryId": str(yt.get("category_id", "28")),
        "privacyStatus": yt.get("privacy_status", "public"),
        "madeForKids": bool(yt.get("made_for_kids", False)),
    }


def _font(size: int) -> ImageFont.FreeTypeFont:
    root = Path(__file__).resolve().parent.parent
    fp = root / CONFIG["visual"]["font"]
    try:
        return ImageFont.truetype(str(fp), size)
    except Exception:  # noqa: BLE001
        return ImageFont.load_default(size)


def make_thumbnail(script: Script, out_path: Path) -> Path:
    tw, th = 1280, 720
    bg = CONFIG["visual"]["background_color"]
    accent = CONFIG["visual"]["accent_color"]

    img = Image.new("RGB", (tw, th), bg)
    draw = ImageDraw.Draw(img)

    # dải accent bên trái
    draw.rectangle([(0, 0), (24, th)], fill=accent)

    title = script.title
    font = _font(84)
    lines = textwrap.wrap(title, width=18)[:4]
    y = th // 2 - (len(lines) * 100) // 2
    for line in lines:
        draw.text((90, y), line, font=font, fill="#e6edf3")
        y += 100

    draw.text((90, th - 90), "TECH • VISUALIZED", font=_font(40), fill=accent)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path
