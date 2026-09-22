"""Render mỗi scene thành 1 ảnh PNG (1920x1080) dựa trên visual_type.

Dùng Pillow cho text/bullets/code/quote/title và Matplotlib cho chart.
Compositor sẽ ghép ảnh + audio thành clip, thêm hiệu ứng zoom nhẹ.

Ghi chú: bản khung dùng ảnh tĩnh cho ổn định trên CI. Có thể nâng cấp
scene "algorithm" sang Manim animation sau (xem manim_scenes.py placeholder).
"""
from __future__ import annotations

import logging
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .config import CONFIG
from .models import Scene

log = logging.getLogger(__name__)

W = CONFIG["visual"]["width"]
H = CONFIG["visual"]["height"]
BG = CONFIG["visual"]["background_color"]
ACCENT = CONFIG["visual"]["accent_color"]
FONT_PATH = CONFIG["visual"]["font"]

_TEXT = "#e6edf3"
_MUTED = "#8b949e"


def _font(size: int) -> ImageFont.FreeTypeFont:
    root = Path(__file__).resolve().parent.parent
    fp = root / FONT_PATH
    try:
        return ImageFont.truetype(str(fp), size)
    except Exception:  # noqa: BLE001 - fallback font mặc định
        return ImageFont.load_default(size)


def _new_canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), BG)
    return img, ImageDraw.Draw(img)


def _draw_center_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    y: int,
    fill: str = _TEXT,
    max_chars: int = 40,
) -> int:
    """Vẽ text căn giữa, tự xuống dòng. Trả về y sau khi vẽ xong."""
    lines = textwrap.wrap(text, width=max_chars) or [""]
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        draw.text(((W - w) / 2, y), line, font=font, fill=fill)
        y += h + 18
    return y


def _render_title(scene: Scene, out: Path) -> None:
    img, draw = _new_canvas()
    # thanh accent trên tiêu đề
    draw.rectangle([(W / 2 - 120, H / 2 - 140), (W / 2 + 120, H / 2 - 128)], fill=ACCENT)
    _draw_center_text(draw, scene.heading or scene.narration[:60], _font(84), H // 2 - 90, max_chars=24)
    img.save(out)


def _render_bullets(scene: Scene, out: Path) -> None:
    img, draw = _new_canvas()
    if scene.heading:
        _draw_center_text(draw, scene.heading, _font(64), 120, fill=ACCENT, max_chars=30)
    y = 340
    bullet_font = _font(46)
    for b in scene.bullets[:5]:
        draw.ellipse([(200, y + 18), (224, y + 42)], fill=ACCENT)
        wrapped = textwrap.wrap(b, width=48) or [""]
        for i, line in enumerate(wrapped):
            draw.text((260, y), line, font=bullet_font, fill=_TEXT)
            y += 62
        y += 24
    img.save(out)


def _render_quote(scene: Scene, out: Path) -> None:
    img, draw = _new_canvas()
    quote = scene.bullets[0] if scene.bullets else scene.narration
    draw.text((W / 2 - 200, H / 2 - 200), "\u201c", font=_font(200), fill=ACCENT)
    _draw_center_text(draw, quote, _font(58), H // 2 - 60, max_chars=34)
    img.save(out)


def _render_code(scene: Scene, out: Path) -> None:
    img, draw = _new_canvas()
    if scene.heading:
        _draw_center_text(draw, scene.heading, _font(56), 90, fill=ACCENT, max_chars=34)
    # panel code
    pad = 160
    draw.rounded_rectangle([(pad, 240), (W - pad, H - 160)], radius=24, fill="#161b22")
    mono = _font(38)
    y = 300
    for line in scene.bullets[:16]:
        draw.text((pad + 60, y), line, font=mono, fill="#c9d1d9")
        y += 52
    img.save(out)


def _render_chart(scene: Scene, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chart = scene.chart or {}
    labels = chart.get("labels", [])
    values = chart.get("values", [])
    kind = chart.get("kind", "bar")

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor(BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG)

    if kind == "line":
        ax.plot(labels, values, color=ACCENT, linewidth=3, marker="o")
    elif kind == "pie":
        ax.pie(values, labels=labels, autopct="%1.0f%%", textprops={"color": _TEXT})
    else:
        ax.bar(labels, values, color=ACCENT)

    if kind != "pie":
        ax.tick_params(colors=_TEXT, labelsize=14)
        for spine in ax.spines.values():
            spine.set_color(_MUTED)
    if scene.heading:
        ax.set_title(scene.heading, color=_TEXT, fontsize=26, pad=20)

    fig.tight_layout()
    fig.savefig(out, facecolor=BG)
    plt.close(fig)


def _render_algorithm(scene: Scene, out: Path) -> None:
    # Bản khung: hiển thị như bullets kèm nhãn thuật toán.
    # Nâng cấp sang Manim animation ở giai đoạn sau.
    fallback = Scene(
        narration=scene.narration,
        visual_type="bullets",
        heading=scene.heading or f"Thuật toán: {scene.algorithm}",
        bullets=scene.bullets or [scene.algorithm],
    )
    _render_bullets(fallback, out)


_RENDERERS = {
    "title": _render_title,
    "bullets": _render_bullets,
    "quote": _render_quote,
    "code": _render_code,
    "chart": _render_chart,
    "algorithm": _render_algorithm,
}


def render_scene(scene: Scene, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    renderer = _RENDERERS.get(scene.visual_type, _render_bullets)
    try:
        renderer(scene, out_path)
    except Exception as e:  # noqa: BLE001 - luôn có ảnh, fallback bullets
        log.warning("Render %s lỗi (%s), fallback bullets", scene.visual_type, e)
        _render_bullets(scene, out_path)
    return out_path
