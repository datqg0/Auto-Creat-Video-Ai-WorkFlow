"""Sinh metadata YouTube (title/description/tags) và thumbnail 1280x720."""
from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .config import CONFIG
from .models import Script

log = logging.getLogger(__name__)


def _fmt_ts(seconds: float) -> str:
    """Định dạng mốc thời gian cho YouTube chapter (m:ss hoặc h:mm:ss)."""
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _build_chapters(script: Script, durations: list[float] | None) -> str:
    """Sinh danh sách chapter (mốc + heading) cho phần mô tả -> SEO + điều hướng.

    YouTube yêu cầu: mốc đầu tiên phải là 0:00 và cần >=3 mốc mới bật chapter.
    """
    if not durations or len(durations) < 3:
        return ""
    lines: list[str] = []
    t = 0.0
    last_label = ""
    for i, scene in enumerate(script.scenes):
        dur = durations[i] if i < len(durations) else 0.0
        label = (scene.heading or "").strip()
        # Bỏ scene không có heading hoặc trùng nhãn liền trước (chapter cần nhãn rõ ràng).
        if label and label != last_label:
            ts = "0:00" if not lines else _fmt_ts(t)
            lines.append(f"{ts} {label}")
            last_label = label
        t += dur
    if len(lines) < 3:
        return ""
    # Ép mốc đầu về 0:00 (yêu cầu bắt buộc của YouTube chapters).
    first = lines[0].split(" ", 1)
    if len(first) == 2 and not first[0].startswith("0:00"):
        lines[0] = "0:00 " + first[1]
    return "Nội dung video:\n" + "\n".join(lines)


def build_metadata(script: Script, durations: list[float] | None = None) -> dict:
    yt = CONFIG["youtube"]
    is_short = CONFIG.get("active_mode") == "short"

    tags = list(dict.fromkeys([*script.tags, *yt.get("default_tags", [])]))
    if is_short and "shorts" not in [t.lower() for t in tags]:
        tags.insert(0, "shorts")
    tags = tags[:15]

    # YouTube nhận diện Short qua #Shorts trong tiêu đề/mô tả (kèm khung hình dọc <60s)
    title = script.title[:100]
    if is_short and "#shorts" not in title.lower():
        title = (title[:90] + " #Shorts")[:100]

    description = script.description.strip()
    if is_short and "#shorts" not in description.lower():
        description = "#Shorts\n\n" + description

    # Chapters (chỉ video dài) -> tăng thời lượng xem + SEO từ khóa heading.
    if not is_short:
        chapters = _build_chapters(script, durations)
        if chapters:
            description += "\n\n" + chapters

    description += "\n\n" + " ".join(f"#{t.replace(' ', '')}" for t in tags[:8])
    description += "\n\nVideo được tạo tự động bằng visualization engine."

    return {
        "title": title,
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

    # Dải gradient tối ở nửa dưới để chữ luôn nổi (kể cả khi đổi màu nền sau này).
    for j in range(th // 2, th):
        a = int(150 * (j - th // 2) / (th / 2))
        draw.line([(0, j), (tw, j)], fill=_blend(bg, "#000000", a / 255))

    # Dải accent bên trái
    draw.rectangle([(0, 0), (24, th)], fill=accent)

    # Chọn cỡ chữ lớn nhất mà tiêu đề vẫn gói gọn <=3 dòng -> chữ to, dễ đọc từ xa.
    title = script.title
    margin = 90
    max_w = tw - margin - 60
    size = 130
    while size > 60:
        font = _font(size)
        lines = _wrap_thumb(draw, title, font, max_w)
        if len(lines) <= 3:
            break
        size -= 8
    else:
        font = _font(size)
        lines = _wrap_thumb(draw, title, font, max_w)[:3]

    line_h = int(size * 1.15)
    total_h = line_h * len(lines)
    y = th - 150 - total_h  # cụm chữ ở nửa dưới
    stroke = max(6, size // 12)
    for line in lines:
        draw.text(
            (margin, y), line, font=font, fill="#ffffff",
            stroke_width=stroke, stroke_fill="#000000",
        )
        y += line_h

    draw.text((margin, th - 70), "TECH • VISUALIZED", font=_font(38), fill=accent)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def _blend(c1: str, c2: str, t: float) -> tuple[int, int, int]:
    """Trộn 2 màu hex theo tỉ lệ t (0..1) -> tuple RGB, cho gradient nền."""
    def rgb(c: str) -> tuple[int, int, int]:
        c = c.lstrip("#")
        return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    a, b = rgb(c1), rgb(c2)
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))  # type: ignore[return-value]


def _wrap_thumb(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    """Gói chữ theo chiều rộng thực đo được (chính xác hơn textwrap theo ký tự)."""
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if draw.textlength(trial, font=font) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines
