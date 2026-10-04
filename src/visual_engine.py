"""Render mỗi scene thành 1 ảnh PNG (1920x1080) dựa trên visual_type.

Dùng Pillow cho text/bullets/code/quote/title/diagram và Matplotlib cho chart.
Compositor sẽ ghép ảnh + audio thành clip, thêm hiệu ứng zoom nhẹ.

Mỗi loại scene có nền gradient + khối trang trí + icon riêng để video sinh
động hơn, không chỉ là chữ trên nền phẳng.
"""
from __future__ import annotations

import hashlib
import logging
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .config import CONFIG
from .models import Scene

log = logging.getLogger(__name__)

def _W() -> int:
    return int(CONFIG["visual"]["width"])


def _H() -> int:
    return int(CONFIG["visual"]["height"])


def _is_vertical() -> bool:
    return _H() > _W()


def _margin() -> int:
    return 80 if _is_vertical() else 120


BG = CONFIG["visual"]["background_color"]
ACCENT = CONFIG["visual"]["accent_color"]
FONT_PATH = CONFIG["visual"]["font"]
FONT_REGULAR = CONFIG["visual"].get("font_regular", FONT_PATH)

_TEXT = "#e6edf3"
_MUTED = "#8b949e"
_PANEL = "#161b22"
# Bảng màu phụ để tô khối trang trí / icon cho đa dạng
_PALETTE = ["#58a6ff", "#3fb950", "#d29922", "#f778ba", "#a371f7", "#39c5cf"]


def _hex(color: str) -> tuple[int, int, int]:
    c = color.lstrip("#")
    return tuple(int(c[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _mix(c1: str, c2: str, t: float) -> tuple[int, int, int]:
    a, b = _hex(c1), _hex(c2)
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))  # type: ignore[return-value]


def _seed(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def _font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    root = Path(__file__).resolve().parent.parent
    fp = root / (FONT_PATH if bold else FONT_REGULAR)
    try:
        return ImageFont.truetype(str(fp), size)
    except Exception:  # noqa: BLE001 - fallback font mặc định
        try:
            return ImageFont.load_default(size)
        except Exception:  # noqa: BLE001
            return ImageFont.load_default()


# ----------------------------- Nền trang trí -----------------------------

def _gradient_bg(top: str = BG, bottom: str = "#010409") -> Image.Image:
    """Nền gradient dọc nhẹ (vẽ theo hàng cho nhanh)."""
    grad = Image.new("RGB", (1, _H()))
    gpx = grad.load()
    for y in range(_H()):
        gpx[0, y] = _mix(top, bottom, y / _H())
    return grad.resize((_W(), _H()))


def _decor_blobs(img: Image.Image, seed: int, count: int = 3) -> Image.Image:
    """Thêm vài khối tròn mờ làm điểm nhấn nền."""
    overlay = Image.new("RGBA", (_W(), _H()), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    rng = seed or 1
    for i in range(count):
        rng = (rng * 1103515245 + 12345) & 0x7FFFFFFF
        cx = rng % _W()
        rng = (rng * 1103515245 + 12345) & 0x7FFFFFFF
        cy = rng % _H()
        rng = (rng * 1103515245 + 12345) & 0x7FFFFFFF
        rad = 180 + (rng % 260)
        r, g, b = _hex(_PALETTE[(seed + i) % len(_PALETTE)])
        od.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=(r, g, b, 34))
    overlay = overlay.filter(ImageFilter.GaussianBlur(80))
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def _load_photo(query: str, index: int = 0) -> Image.Image | None:
    """Tải ảnh gốc theo từ khóa (index để lấy ảnh khác nhau, đa dạng hình)."""
    try:
        from .image_fetcher import fetch_image

        path = fetch_image(query, index=index)
        if not path:
            return None
        return Image.open(path).convert("RGB")
    except Exception as e:  # noqa: BLE001
        log.debug("Tải ảnh lỗi: %s", e)
        return None


def _cover(photo: Image.Image, w: int, h: int) -> Image.Image:
    """Scale + crop ảnh phủ kín khung wxh (giữ tỉ lệ)."""
    scale = max(w / photo.width, h / photo.height)
    photo = photo.resize((max(1, int(photo.width * scale)), max(1, int(photo.height * scale))))
    left = (photo.width - w) // 2
    top = (photo.height - h) // 2
    return photo.crop((left, top, left + w, top + h))


def _photo_bg(query: str, index: int = 0) -> Image.Image | None:
    """Ảnh minh họa làm nền: crop full khung, blur nhẹ + phủ tối để chữ đọc rõ."""
    photo = _load_photo(query, index)
    if photo is None:
        return None
    photo = _cover(photo, _W(), _H()).filter(ImageFilter.GaussianBlur(6))
    overlay = Image.new("RGBA", (_W(), _H()), (5, 8, 16, 190))
    return Image.alpha_composite(photo.convert("RGBA"), overlay).convert("RGB")


def _photo_panel(
    img: Image.Image, query: str, box: tuple[int, int, int, int], index: int = 0, radius: int = 24
) -> bool:
    """Dán 1 ảnh minh họa RÕ NÉT vào vùng box (bo góc + viền). True nếu dán được."""
    photo = _load_photo(query, index)
    if photo is None:
        return False
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    thumb = _cover(photo, w, h)
    # mask bo góc
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w, h], radius=radius, fill=255)
    img.paste(thumb, (x0, y0), mask)
    # viền accent nhẹ
    ImageDraw.Draw(img).rounded_rectangle(
        [x0, y0, x1, y1], radius=radius, outline=ACCENT, width=3
    )
    return True


_IMAGES_ON = CONFIG.get("images", {}).get("enabled", False)
# Khung dọc (short 9:16) và lề an toàn được tính động qua _is_vertical() và _margin()


def _new_canvas(
    seed: int = 0, blobs: int = 3, image_query: str = ""
) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = _photo_bg(image_query) if (image_query and _IMAGES_ON) else None
    if img is None:
        img = _gradient_bg()
        if blobs:
            img = _decor_blobs(img, seed, blobs)
    return img, ImageDraw.Draw(img)


def _footer(draw: ImageDraw.ImageDraw) -> None:
    draw.line([(120, _H() - 90), (_W() - 120, _H() - 90)], fill=_MUTED, width=2)
    draw.ellipse([(120, _H() - 78), (140, _H() - 58)], fill=ACCENT)


def _text_w(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def _draw_center_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    y: int,
    fill: str = _TEXT,
    max_chars: int = 40,
) -> int:
    """Vẽ text căn giữa, tự xuống dòng theo BỀ RỘNG PIXEL. Trả về y sau khi vẽ."""
    max_w = _W() - 2 * _margin()
    lines = _wrap_lines(draw, text, font, max_w) or [""]
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        draw.text(((_W() - w) / 2, y), line, font=font, fill=fill)
        y += h + 18
    return y


def _fit_font(
    draw: ImageDraw.ImageDraw, text: str, base_size: int, max_w: int, min_size: int = 28, bold: bool = True
) -> ImageFont.FreeTypeFont:
    """Chọn cỡ chữ lớn nhất mà từ dài nhất vẫn vừa bề rộng max_w (chống tràn khung)."""
    longest = max(text.split(), key=len) if text.split() else text
    size = base_size
    while size > min_size:
        f = _font(size, bold=bold)
        if _text_w(draw, longest, f) <= max_w:
            return f
        size -= 4
    return _font(min_size, bold=bold)


def _icon(draw: ImageDraw.ImageDraw, x: int, y: int, size: int, color: str, idx: int) -> None:
    """Vẽ icon hình học nhỏ cạnh mỗi bullet cho sinh động."""
    r, g, b = _hex(color)
    shape = idx % 4
    if shape == 0:
        draw.ellipse([x, y, x + size, y + size], fill=(r, g, b))
    elif shape == 1:
        draw.rounded_rectangle([x, y, x + size, y + size], radius=6, fill=(r, g, b))
    elif shape == 2:
        draw.polygon([(x + size // 2, y), (x, y + size), (x + size, y + size)], fill=(r, g, b))
    else:
        draw.polygon(
            [(x + size // 2, y), (x + size, y + size // 2), (x + size // 2, y + size), (x, y + size // 2)],
            fill=(r, g, b),
        )


# ----------------------------- Renderers -----------------------------

def _render_title(scene: Scene, out: Path) -> None:
    heading = scene.heading or scene.narration[:60]
    img, draw = _new_canvas(_seed(heading), blobs=4, image_query=scene.image_query)
    cx, cy = _W() // 2, _H() // 2 - 20
    # vòng tròn đồng tâm trang trí (thu nhỏ theo khung dọc để không tràn ngang)
    base_r = min(_W(), _H()) // 3
    rings = (base_r, int(base_r * 0.78), int(base_r * 0.56))
    for i, rad in enumerate(rings):
        r, g, b = _hex(_PALETTE[i % len(_PALETTE)])
        ring = Image.new("RGBA", (_W(), _H()), (0, 0, 0, 0))
        ImageDraw.Draw(ring).ellipse(
            [cx - rad, cy - rad, cx + rad, cy + rad], outline=(r, g, b, 90), width=3
        )
        img = Image.alpha_composite(img.convert("RGBA"), ring).convert("RGB")

    # Tìm icon công nghệ phù hợp với chủ đề để làm visual đặt vấn đề
    try:
        import re
        from .asset_manager import get_tech_icon, icon_to_png
        words = re.findall(r"[A-Za-z0-9\+\#\.\-]+", f"{heading} {scene.narration[:120]}")
        for w in words:
            if len(w) >= 2 and w.lower() not in ("la", "va", "co", "trong", "cho", "cac", "the", "and", "how", "what", "tai", "sao"):
                svg = get_tech_icon(w, color_hex="38bdf8")
                if svg:
                    png_path = out.parent / f"_icon_{svg.stem}.png"
                    png = icon_to_png(svg, png_path, size=140)
                    if png and png.exists():
                        with Image.open(png) as icon_img:
                            glow = Image.new("RGBA", (_W(), _H()), (0, 0, 0, 0))
                            ix, iy = cx - 70, cy - 240
                            ImageDraw.Draw(glow).ellipse([cx - 85, cy - 255, cx + 85, cy - 85], fill=(56, 189, 248, 60))
                            img = Image.alpha_composite(img.convert("RGBA"), glow).convert("RGB")
                            img.paste(icon_img.convert("RGBA"), (ix, iy), icon_img.convert("RGBA"))
                            break
    except Exception as e:
        log.debug("Chèn icon visual hook lỗi: %s", e)

    draw = ImageDraw.Draw(img)
    draw.rectangle([(cx - 140, cy - 110), (cx + 140, cy - 100)], fill=ACCENT)
    # Cỡ chữ tự co để từ dài nhất vừa bề rộng khung -> không tràn (nhất là short).
    hfont = _fit_font(draw, heading, 96 if _is_vertical() else 80, _W() - 2 * _margin())
    _draw_center_text(draw, heading, hfont, cy - 70)
    _footer(draw)
    img.save(out)


def _render_challenge(scene: Scene, out: Path) -> None:
    """Render card bài toán / thử thách trực quan có icon và visual card bắt mắt."""
    img, draw = _new_canvas(_seed(scene.heading or scene.narration), blobs=3, image_query=scene.image_query)

    cw, ch = int(_W() * 0.85), int(_H() * 0.72)
    cx0, cy0 = (_W() - cw) // 2, (_H() - ch) // 2 - 10
    cx1, cy1 = cx0 + cw, cy0 + ch

    # Panel bo góc nổi bật
    panel = Image.new("RGBA", (_W(), _H()), (0, 0, 0, 0))
    pdraw = ImageDraw.Draw(panel)
    pdraw.rounded_rectangle([cx0, cy0, cx1, cy1], radius=24, fill=(18, 24, 38, 230), outline=(245, 158, 11, 220), width=3)
    img = Image.alpha_composite(img.convert("RGBA"), panel).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Badge: 🎯 THỬ THÁCH VẬN DỤNG
    badge_w, badge_h = 360, 56
    bx0, by0 = (_W() - badge_w) // 2, cy0 - 28
    draw.rounded_rectangle([bx0, by0, bx0 + badge_w, by0 + badge_h], radius=28, fill="#f59e0b", outline="#ffffff", width=2)
    bfont = _font(26, bold=True)
    draw.text((bx0 + 36, by0 + 14), "🎯 THỬ THÁCH VẬN DỤNG", font=bfont, fill="#000000")

    # Icon dấu hỏi tròn
    icon_w = 60
    draw.ellipse([cx0 + 60, cy0 + 60, cx0 + 60 + icon_w, cy0 + 60 + icon_w], fill="#fbbf24")
    draw.text((cx0 + 80, cy0 + 66), "?", font=_font(42, bold=True), fill="#000000")

    # Câu hỏi
    q_font = _font(40 if not _is_vertical() else 34, bold=True)
    bullets = scene.bullets or [scene.narration]
    question_text = bullets[0] if bullets else "Hãy suy nghĩ giải pháp cho bài toán này."

    qx = cx0 + 150
    qy = cy0 + 65
    max_qw = cx1 - qx - 60
    q_lines = _wrap_lines(draw, question_text, q_font, max_qw)
    for ql in q_lines[:4]:
        draw.text((qx, qy), ql, font=q_font, fill="#ffffff")
        qy += 56

    # Hộp gợi ý (Hint box) nếu có
    if len(bullets) > 1 and bullets[1]:
        hint_text = bullets[1]
        hy0 = max(qy + 35, cy0 + ch - 170)
        draw.rounded_rectangle([cx0 + 50, hy0, cx1 - 50, hy0 + 95], radius=16, fill="#0d1117", outline="#38bdf8", width=2)
        h_font = _font(28, bold=False)
        draw.text((cx0 + 75, hy0 + 26), "💡 " + hint_text, font=h_font, fill="#38bdf8")

    # Call to action ở đáy
    cta_font = _font(24, bold=True)
    draw.text(((_W() - 520) // 2, cy1 - 42), "💬 ĐỂ LẠI ĐÁP ÁN CỦA BẠN DƯỚI PHẦN BÌNH LUẬN", font=cta_font, fill="#f59e0b")

    _footer(draw)
    img.save(out)


def _render_bullets(scene: Scene, out: Path) -> None:
    if _is_vertical():
        _render_bullets_vertical(scene, out)
        return
    # Bố cục 2 cột: chữ bên trái, ảnh minh họa RÕ NÉT bên phải (nếu có ảnh)
    img = _gradient_bg()
    img = _decor_blobs(img, _seed(scene.heading or scene.narration), 3)
    draw = ImageDraw.Draw(img)

    panel_placed = False
    text_right = _W() - 160
    if _IMAGES_ON and scene.image_query:
        pw, ph = 620, 620
        px0 = _W() - 160 - pw
        py0 = (_H() - ph) // 2 + 20
        panel_placed = _photo_panel(img, scene.image_query, (px0, py0, px0 + pw, py0 + ph))
        draw = ImageDraw.Draw(img)
        if panel_placed:
            text_right = px0 - 60

    text_left = 170
    text_w = text_right - text_left
    if scene.heading:
        draw.rectangle([(120, 120), (132, 200)], fill=ACCENT)
        hfont = _fit_font(draw, scene.heading, 60, text_w)
        for j, hl in enumerate(_wrap_lines(draw, scene.heading, hfont, text_w)[:3]):
            draw.text((text_left, 120 + j * (hfont.size + 8)), hl, font=hfont, fill=_TEXT)
    y = 320
    bullet_font = _font(44, bold=False)
    for i, b in enumerate(scene.bullets[:5]):
        color = _PALETTE[i % len(_PALETTE)]
        _icon(draw, 190, y + 8, 34, color, i)
        for line in _wrap_lines(draw, b, bullet_font, text_right - 260):
            draw.text((260, y), line, font=bullet_font, fill=_TEXT)
            y += 58
        y += 26
    _footer(draw)
    img.save(out)


def _render_bullets_vertical(scene: Scene, out: Path) -> None:
    """Bố cục DỌC cho Short: heading trên, ảnh giữa, bullets dưới — chữ to, không tràn."""
    img = _gradient_bg()
    img = _decor_blobs(img, _seed(scene.heading or scene.narration), 3)
    draw = ImageDraw.Draw(img)
    inner_w = _W() - 2 * _margin()

    y = 180
    if scene.heading:
        draw.rectangle([(_margin(), y), (_margin() + 14, y + 90)], fill=ACCENT)
        hfont = _fit_font(draw, scene.heading, 88, inner_w - 40)
        for hl in _wrap_lines(draw, scene.heading, hfont, inner_w - 40)[:3]:
            draw.text((_margin() + 40, y), hl, font=hfont, fill=_TEXT)
            y += hfont.size + 12
        y += 40

    # Ảnh minh họa vuông ở giữa
    if _IMAGES_ON and scene.image_query:
        side = min(inner_w, 760)
        px0 = (_W() - side) // 2
        if _photo_panel(img, scene.image_query, (px0, y, px0 + side, y + side)):
            draw = ImageDraw.Draw(img)
            y += side + 60

    bullet_font = _font(52, bold=False)
    for i, b in enumerate(scene.bullets[:4]):
        color = _PALETTE[i % len(_PALETTE)]
        _icon(draw, _margin(), y + 8, 40, color, i)
        for line in _wrap_lines(draw, b, bullet_font, inner_w - 80):
            draw.text((_margin() + 70, y), line, font=bullet_font, fill=_TEXT)
            y += 68
        y += 30
    img.save(out)


def _render_quote(scene: Scene, out: Path) -> None:
    quote = scene.bullets[0] if scene.bullets else scene.narration
    img, draw = _new_canvas(_seed(quote), blobs=4, image_query=scene.image_query)
    qfont = _font(200 if _is_vertical() else 240)
    draw.text((_margin(), _H() / 2 - (300 if _is_vertical() else 240)), "\u201c", font=qfont, fill=ACCENT)
    body = _fit_font(draw, quote, 64 if _is_vertical() else 56, _W() - 2 * _margin(), bold=False)
    _draw_center_text(draw, quote, body, _H() // 2 - 40)
    _footer(draw)
    img.save(out)


def _render_code(scene: Scene, out: Path) -> None:
    img, draw = _new_canvas(_seed(scene.heading or "code"), blobs=2)
    if scene.heading:
        draw.rectangle([(_margin(), 90), (_margin() + 12, 160)], fill=ACCENT)
        hfont = _fit_font(draw, scene.heading, 52, _W() - 2 * _margin() - 60)
        draw.text((_margin() + 50, 96), scene.heading, font=hfont, fill=_TEXT)
    pad = _margin()
    top = 220
    draw.rounded_rectangle([(pad, top), (_W() - pad, _H() - 150)], radius=20, fill=_PANEL)
    draw.rounded_rectangle([(pad, top), (_W() - pad, top + 46)], radius=20, fill="#21262d")
    for k, dot in enumerate(("#ff5f56", "#ffbd2e", "#27c93f")):
        r, g, b = _hex(dot)
        draw.ellipse([pad + 24 + k * 34, top + 16, pad + 40 + k * 34, top + 32], fill=(r, g, b))
    mono = _font(30 if _is_vertical() else 34, bold=False)
    y = top + 78
    for line in scene.bullets[:20]:
        draw.text((pad + 40, y), line, font=mono, fill="#c9d1d9")
        y += 46
    _footer(draw)
    img.save(out)


def _render_chart(scene: Scene, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    root = Path(__file__).resolve().parent.parent
    try:
        fp = font_manager.FontProperties(fname=str(root / FONT_PATH))
    except Exception:  # noqa: BLE001
        fp = None

    chart = scene.chart or {}
    labels = chart.get("labels", [])
    values = chart.get("values", [])
    kind = chart.get("kind", "bar")

    fig = plt.figure(figsize=(_W() / 100, _H() / 100), dpi=100)
    fig.patch.set_facecolor(BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG)
    colors = _PALETTE * (len(values) // len(_PALETTE) + 1)

    if kind == "line":
        ax.plot(labels, values, color=ACCENT, linewidth=4, marker="o", markersize=9)
        ax.fill_between(range(len(values)), values, alpha=0.15, color=ACCENT)
    elif kind == "pie":
        tp = {"color": _TEXT}
        if fp:
            tp["fontproperties"] = fp
        ax.pie(values, labels=labels, autopct="%1.0f%%", colors=colors[: len(values)], textprops=tp)
    else:
        ax.bar(labels, values, color=colors[: len(values)])

    if kind != "pie":
        ax.tick_params(colors=_TEXT, labelsize=16)
        for spine in ax.spines.values():
            spine.set_color(_MUTED)
        if fp:
            for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                lbl.set_fontproperties(fp)
    if scene.heading:
        ax.set_title(scene.heading, color=_TEXT, fontsize=30, pad=24, fontproperties=fp)

    fig.tight_layout()
    fig.savefig(out, facecolor=BG)
    plt.close(fig)


def _render_diagram(scene: Scene, out: Path) -> None:
    """Sơ đồ luồng: các bước nối bằng mũi tên (dùng bullets làm node)."""
    img, draw = _new_canvas(_seed(scene.heading or scene.algorithm or "diagram"), blobs=2, image_query=scene.image_query)
    if scene.heading:
        _draw_center_text(draw, scene.heading, _font(56), 90, fill=_TEXT, max_chars=34)

    steps = scene.bullets[:5] or [scene.algorithm or "Bước"]
    n = len(steps)
    gap = 52
    box_w = min(760, _W() - 2 * _margin())  # co theo khung, không tràn ngang (short)
    cx = _W() // 2
    node_font = _font(36, bold=False)
    line_spacing = 8
    pad_v = 26  # đệm trên/dưới trong box
    text_max_w = box_w - 130  # trừ chỗ badge số + lề trong

    # Tính chiều cao từng box theo số dòng chữ (chữ dài -> box cao hơn)
    wrapped: list[str] = []
    box_hs: list[int] = []
    ascent, descent = node_font.getmetrics()
    line_h = ascent + descent
    for step in steps:
        text = "\n".join(_wrap_lines(draw, step, node_font, text_max_w))
        wrapped.append(text)
        n_lines = text.count("\n") + 1
        h = pad_v * 2 + n_lines * line_h + (n_lines - 1) * line_spacing
        box_hs.append(max(110, h))

    total_h = sum(box_hs) + (n - 1) * gap
    y = max((_H() - total_h) // 2 + 30, 200)

    for i, (step, box_h) in enumerate(zip(wrapped, box_hs)):
        r, g, b = _hex(_PALETTE[i % len(_PALETTE)])
        x0 = cx - box_w // 2
        mid = y + box_h // 2
        draw.rounded_rectangle(
            [x0, y, x0 + box_w, y + box_h], radius=18, outline=(r, g, b), width=4, fill=_PANEL
        )
        # badge số thứ tự, căn giữa dọc
        draw.ellipse([x0 + 22, mid - 24, x0 + 70, mid + 24], fill=(r, g, b))
        num = str(i + 1)
        nw = _text_w(draw, num, _font(34))
        draw.text((x0 + 46 - nw / 2, mid - 22), num, font=_font(34), fill="#0d1117")
        # chữ căn giữa dọc trong box
        draw.multiline_text(
            (x0 + 96, mid),
            step,
            font=node_font,
            fill=_TEXT,
            spacing=line_spacing,
            anchor="lm",
        )
        if i < n - 1:
            ay = y + box_h
            draw.line([(cx, ay), (cx, ay + gap)], fill=ACCENT, width=4)
            draw.polygon(
                [(cx - 12, ay + gap - 14), (cx + 12, ay + gap - 14), (cx, ay + gap)], fill=ACCENT
            )
        y += box_h + gap
    _footer(draw)
    img.save(out)


def _render_algorithm(scene: Scene, out: Path) -> None:
    # Thuật toán -> vẽ dạng sơ đồ luồng cho trực quan
    if scene.bullets:
        _render_diagram(scene, out)
    else:
        fallback = Scene(
            narration=scene.narration,
            visual_type="bullets",
            heading=scene.heading or f"Thuật toán: {scene.algorithm}",
            bullets=[scene.algorithm or scene.narration],
        )
        _render_bullets(fallback, out)


_RENDERERS = {
    "title": _render_title,
    "bullets": _render_bullets,
    "quote": _render_quote,
    "code": _render_code,
    "chart": _render_chart,
    "algorithm": _render_algorithm,
    "diagram": _render_diagram,
    "challenge": _render_challenge,
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


# --------------------------- Overlay cho video b-roll ---------------------------

def _wrap_lines(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    """Bọc chữ theo bề rộng pixel (chính xác hơn textwrap theo ký tự)."""
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if _text_w(draw, trial, font) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def render_overlay(scene: Scene, out_path: Path) -> Path:
    """Render lớp phủ TRONG SUỐT (RGBA PNG) để đặt lên video b-roll.

    Gồm: scrim tối ở trên/dưới cho dễ đọc, heading góc trên, và caption (bullets
    hoặc câu narration rút gọn) ở dưới. Nền trong suốt -> lộ video phía sau.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGBA", (_W(), _H()), (0, 0, 0, 0))

    # Scrim tối ở trên và dưới để chữ nổi trên video
    scrim = Image.new("RGBA", (_W(), _H()), (0, 0, 0, 0))
    sd = ImageDraw.Draw(scrim)
    top_h = int(_H() * 0.30)
    bot_h = int(_H() * 0.42)
    for y in range(top_h):
        a = int(150 * (1 - y / top_h))
        sd.line([(0, y), (_W(), y)], fill=(5, 8, 16, a))
    for y in range(_H() - bot_h, _H()):
        a = int(190 * ((y - (_H() - bot_h)) / bot_h))
        sd.line([(0, y), (_W(), y)], fill=(5, 8, 16, a))
    img = Image.alpha_composite(img, scrim)
    draw = ImageDraw.Draw(img)

    margin = 90 if _W() >= 1600 else 70
    is_vertical = _H() > _W()

    heading = scene.heading or ""
    if heading:
        hfont = _font(72 if not is_vertical else 76)
        draw.rectangle([(margin, margin), (margin + 12, margin + 84)], fill=ACCENT)
        for i, line in enumerate(_wrap_lines(draw, heading, hfont, _W() - 2 * margin - 40)[:3]):
            draw.text((margin + 34, margin + i * 90), line, font=hfont, fill=_TEXT)

    # Chỉ vẽ caption khi có bullets (ý chính); narration đã hiện ở phụ đề burn-in
    # nên KHÔNG vẽ lại để tránh trùng/chồng chữ.
    if scene.bullets:
        caption_font = _font(52 if not is_vertical else 60, bold=False)
        lines: list[str] = []
        for b in scene.bullets[:4]:
            lines.extend(_wrap_lines(draw, "• " + b, caption_font, _W() - 2 * margin))
        lines = lines[:6]
        asc, desc = caption_font.getmetrics()
        line_h = asc + desc + 16
        total = len(lines) * line_h
        # Phụ đề burn-in nằm ở ĐÁY màn hình (ffmpeg subtitles, FontSize=22,
        # BorderStyle=3). Dành riêng ~18% chiều cao đáy cho phụ đề và ĐẶT
        # caption NẰM TRÊN vùng đó -> chữ overlay luôn ở trên phụ đề, không
        # còn chồng lên nhau dù phụ đề xuất hiện giữa scene.
        sub_reserve = int(_H() * 0.18)
        bottom_limit = _H() - sub_reserve
        y = bottom_limit - total
        # Tránh đè lên heading ở góc trên khi caption quá dài.
        y = max(y, top_h + 20)
        for line in lines:
            draw.text((margin, y), line, font=caption_font, fill=_TEXT)
            y += line_h

    img.save(out_path)
    return out_path
