"""Thumbnail phong cách Kurzgesagt: AI sinh NỀN minh họa + PIL overlay chữ Việt.

Bố cục chuẩn YouTube Tech:
- Bên trái (40%): Chữ tiêu đề to bản, rõ ràng, căn giữa trên nền tối màu chủ đạo.
- Ở giữa (5%): Dải gradient alpha chuyển mượt mà (smoothstep), hòa quyện ảnh vào nền.
- Bên phải (60%): Ảnh AI minh họa phong cách Kurzgesagt (FLUX / Pollinations / Gemini).

Không có key / lỗi API -> trả None để caller fallback về SVG hoặc make_thumbnail.
"""
from __future__ import annotations

import io
import logging
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .config import CONFIG, env
from .models import Script

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent


def _font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    key = "font" if bold else "font_regular"
    fp = _ROOT / CONFIG["visual"][key]
    try:
        return ImageFont.truetype(str(fp), size)
    except Exception:  # noqa: BLE001
        return ImageFont.load_default(size)


def _visual_brief(script: Script) -> str:
    """Nhờ LLM đọc nội dung video -> mô tả cảnh minh họa (tiếng Anh) cho FLUX.

    FLUX hiểu tiếng Việt kém nên hay vẽ lạc đề; ta để LLM chọn sẵn các vật thể/
    biểu tượng cụ thể bám chủ đề rồi mới đưa vào prompt sinh ảnh.
    """
    subject = script.topic or script.title
    context = subject
    if script.scenes:
        headings = [s.heading for s in script.scenes[:4] if s.heading]
        if headings:
            context += ". Nội dung chính: " + "; ".join(headings)

    competitor_context = ""
    try:
        from .tinyfish_client import is_available, research_youtube_competitors

        if is_available():
            comp_data = research_youtube_competitors(subject, limit=3)
            if comp_data and "competitors" in comp_data:
                titles = [c["title"] for c in comp_data["competitors"] if c.get("title")]
                if titles:
                    competitor_context = "\nTham khảo top video YouTube thành công cùng chủ đề:\n" + "\n".join(f"- {t}" for t in titles)
    except Exception as e:  # noqa: BLE001
        log.debug("TinyFish competitor research loi: %s", e)

    try:
        from .llm import generate

        out = generate(
            prompt=(
                "Video tiếng Việt sau đây cần một thumbnail minh họa.\n"
                f"Chủ đề: {context}\n{competitor_context}\n\n"
                "Hãy mô tả BẰNG TIẾNG ANH (1-2 câu, tối đa 40 từ) một cảnh minh họa "
                "cụ thể, bám sát chủ đề: liệt kê các VẬT THỂ/BIỂU TƯỢNG chính nên "
                "xuất hiện (ví dụ mạch điện, khóa, nơ-ron, gói tin...). "
                "KHÔNG mô tả phong cách, KHÔNG nhắc chữ/text. Chỉ tả nội dung cảnh."
            ),
            system="You are an art director. Reply with a concise English scene description only.",
        )
        brief = " ".join(out.split()).strip().strip('"')
        if brief:
            return brief
    except Exception as e:  # noqa: BLE001 - lỗi LLM -> dùng subject thô
        log.warning("Thumbnail AI: LLM mô tả cảnh lỗi (%s) -> dùng chủ đề thô", e)
    return subject


def _build_bg_prompt(script: Script) -> str:
    """Prompt tả ảnh minh họa Kurzgesagt căn giữa, viền mềm tối, không chữ."""
    brief = _visual_brief(script)
    return (
        f"Flat vector illustration in Kurzgesagt art style. Scene: {brief}. "
        "Bold saturated colors, smooth gradients, clean flat 2D shapes, soft glow, "
        "cinematic lighting, conceptual editorial illustration. "
        "Main subject centered in the frame, fills the scene, dark soft edges, no text, no watermark, no words."
    )


def _fit_cover(img: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    """Cắt trung tâm và co dãn ảnh theo chuẩn tỉ lệ khung hình đích (cover mode)."""
    target_w, target_h = target_size
    img = img.convert("RGB")
    w, h = img.size

    # Phóng nhẹ + cắt bỏ biên viền (khoảng 2%) mà model ảnh hay để lại ở mép
    inset = int(min(w, h) * 0.02)
    if inset > 0:
        img = img.crop((inset, inset, w - inset, h - inset))
        w, h = img.size

    target_ratio = target_w / target_h
    current_ratio = w / h

    if current_ratio > target_ratio:
        # Ảnh quá rộng -> cắt bớt 2 bên trái phải theo tâm
        new_w = int(h * target_ratio)
        left = max(0, (w - new_w) // 2)
        img = img.crop((left, 0, left + new_w, h))
    else:
        # Ảnh quá cao -> cắt bớt trên dưới (lệch xuống chút để giữ chủ thể)
        new_h = int(w / target_ratio)
        top = max(0, int((h - new_h) * 0.38))
        img = img.crop((0, top, w, top + new_h))

    return img.resize(target_size, Image.Resampling.LANCZOS)


def _dominant_dark_color(img: Image.Image) -> tuple[int, int, int]:
    """Lấy màu chủ đạo từ mép trái của ảnh và làm tối sâu để tạo nền cột chữ ăn màu hoàn hảo."""
    w, h = img.size
    # Sample dải 25% phía bên trái của ảnh (tiếp giáp trực tiếp với dải gradient)
    left_strip = img.crop((0, 0, max(10, int(w * 0.25)), h))
    small = left_strip.resize((1, 1), Image.Resampling.BILINEAR)
    r, g, b = small.getpixel((0, 0))[:3]
    # Giảm độ sáng về mức tối sâu (khoảng 20%) để chữ trắng nổi bật tuyệt đối
    factor = 0.20
    r_dark = max(8, min(40, int(r * factor)))
    g_dark = max(10, min(45, int(g * factor)))
    b_dark = max(16, min(55, int(b * factor)))
    return (r_dark, g_dark, b_dark)


def _gemini_background(script: Script, size: tuple[int, int]) -> Image.Image | None:
    api_key = env("GEMINI_API_KEY")
    if not api_key:
        log.info("Thumbnail AI: thiếu GEMINI_API_KEY -> bỏ qua")
        return None

    model = CONFIG.get("thumbnail", {}).get("model", "gemini-2.5-flash-image")
    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=model,
            contents=_build_bg_prompt(script),
        )
        for part in resp.candidates[0].content.parts:
            inline = getattr(part, "inline_data", None)
            if inline and inline.data:
                img = Image.open(io.BytesIO(inline.data)).convert("RGB")
                return _fit_cover(img, size)
        log.warning("Thumbnail AI: phản hồi không có ảnh")
        return None
    except Exception as e:  # noqa: BLE001 - lỗi API -> fallback
        log.warning("Thumbnail AI (Gemini) lỗi (%s)", e)
        return None


def _huggingface_background(script: Script, size: tuple[int, int]) -> Image.Image | None:
    """Sinh ảnh qua HuggingFace Inference (FLUX.1-schnell). Cần HF_TOKEN."""
    token = env("HF_TOKEN")
    if not token:
        log.info("Thumbnail AI: thiếu HF_TOKEN -> bỏ qua HuggingFace")
        return None

    model = CONFIG.get("thumbnail", {}).get("hf_model", "black-forest-labs/FLUX.1-schnell")
    try:
        from huggingface_hub import InferenceClient

        client = InferenceClient(api_key=token)
        prompt = _build_bg_prompt(script)
        try:
            img = client.text_to_image(prompt, model=model, width=size[0], height=size[1])
        except Exception:
            img = client.text_to_image(prompt, model=model)
        return _fit_cover(img.convert("RGB"), size)
    except Exception as e:  # noqa: BLE001 - lỗi -> fallback tiếp
        log.warning("Thumbnail AI (HuggingFace) lỗi (%s)", e)
        return None


def _pollinations_background(script: Script, size: tuple[int, int]) -> Image.Image | None:
    """Fallback sinh ảnh KHÔNG cần key qua Pollinations (Flux)."""
    prompt = _build_bg_prompt(script)
    url = (
        "https://image.pollinations.ai/prompt/"
        + urllib.parse.quote(prompt)
        + f"?width={size[0]}&height={size[1]}&nologo=true&model=flux"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=90) as r:  # noqa: S310 - URL cố định
            data = r.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return _fit_cover(img, size)
    except Exception as e:  # noqa: BLE001 - lỗi -> fallback tiếp
        log.warning("Thumbnail AI (Pollinations) lỗi (%s)", e)
        return None


def _generate_background(script: Script, size: tuple[int, int]) -> Image.Image | None:
    """Chuỗi ưu tiên: HuggingFace (FLUX) -> Pollinations -> Gemini (nếu có billing)."""
    bg = _huggingface_background(script, size)
    if bg is not None:
        return bg
    if CONFIG.get("thumbnail", {}).get("pollinations_fallback", True):
        log.info("Thumbnail AI: fallback sang Pollinations (không cần key)")
        bg = _pollinations_background(script, size)
        if bg is not None:
            return bg
    if CONFIG.get("thumbnail", {}).get("gemini_fallback", False):
        return _gemini_background(script, size)
    return None


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
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


def _compose_split(
    img: Image.Image,
    canvas_size: tuple[int, int] = (1280, 720),
    text_ratio: float = 0.40,
    grad_ratio: float = 0.05,
) -> tuple[Image.Image, int]:
    """Ghép ảnh vào nửa phải, tạo dải gradient alpha chuyển mượt sang nền tối bên trái."""
    W, H = canvas_size
    text_w = int(W * text_ratio)          # 512px
    grad_w = max(16, int(W * grad_ratio)) # 64px
    img_w = W - text_w + grad_w           # 832px
    img_x = text_w - grad_w               # 448px

    fitted_img = _fit_cover(img, (img_w, H))
    bg_color = _dominant_dark_color(fitted_img)

    canvas = Image.new("RGBA", (W, H), (*bg_color, 255))

    # Tạo mask alpha cho dải gradient bằng công thức smoothstep: 3*t^2 - 2*t^3
    mask = Image.new("L", (img_w, H), 255)
    mask_draw = ImageDraw.Draw(mask)
    for x in range(grad_w):
        t = x / float(grad_w)
        alpha = int(255 * (3 * t**2 - 2 * t**3))
        mask_draw.line([(x, 0), (x, H)], fill=alpha)

    rgba_img = fitted_img.convert("RGBA")
    rgba_img.putalpha(mask)
    canvas.alpha_composite(rgba_img, (img_x, 0))

    return canvas, text_w


def _draw_split_title(canvas: Image.Image, script: Script, text_w: int) -> Image.Image:
    """Vẽ cụm tiêu đề to bản, sắc nét ở cột bên trái (0 -> text_w)."""
    W, H = canvas.size
    draw = ImageDraw.Draw(canvas)

    title = (script.title or script.topic).upper().strip()
    pad_left = int(W * 0.04)               # ~51px
    pad_right = int(text_w * 0.08)         # ~41px
    max_text_w = text_w - pad_left - pad_right # ~420px

    # Tự động co cỡ chữ để vừa trong tối đa 4 dòng
    size = int(H * 0.088)                  # bắt đầu từ ~63px
    font = _font(size, bold=True)
    lines = _wrap(draw, title, font, max_text_w)
    while len(lines) > 4 and size > int(H * 0.055):
        size -= 4
        font = _font(size, bold=True)
        lines = _wrap(draw, title, font, max_text_w)

    line_h = int(size * 1.18)
    badge_h = 36
    spacing = 16
    total_h = badge_h + spacing + line_h * len(lines)
    start_y = max(40, (H - total_h) // 2)

    # 1. Badge pill chủ đề phía trên tiêu đề
    badge_font = _font(18, bold=True)
    badge_text = "KIẾN THỨC CÔNG NGHỆ"
    bw = int(draw.textlength(badge_text, font=badge_font)) + 26
    badge_rect = [pad_left, start_y, pad_left + bw, start_y + badge_h]
    draw.rounded_rectangle(badge_rect, radius=8, fill=(15, 23, 42, 220), outline="#38bdf8", width=2)
    draw.text((pad_left + 13, start_y + 8), badge_text, font=badge_font, fill="#38bdf8")

    # 2. Tiêu đề chính: màu trắng, viền đen dày dặn nổi bật
    text_y = start_y + badge_h + spacing
    stroke = max(5, size // 10)
    for i, line in enumerate(lines[:4]):
        y = text_y + i * line_h
        draw.text(
            (pad_left, y),
            line,
            font=font,
            fill="#ffffff",
            stroke_width=stroke,
            stroke_fill="#000000",
        )

    return canvas.convert("RGB")


def make_ai_thumbnail(script: Script, out_path: Path) -> Path | None:
    """Sinh thumbnail bố cục chia đôi (chữ trái 40%, gradient chuyển màu 5%, ảnh AI phải 60%)."""
    cfg = CONFIG.get("thumbnail", {})
    canvas_w, canvas_h = cfg.get("canvas", [1280, 720])
    text_ratio = float(cfg.get("text_ratio", 0.40))
    grad_ratio = float(cfg.get("gradient_ratio", 0.05))

    # Tính kích thước ảnh cần sinh cho khung bên phải (bội số của 16)
    grad_w = max(16, int(canvas_w * grad_ratio))
    target_img_w = ((canvas_w - int(canvas_w * text_ratio) + grad_w) // 16) * 16  # 832
    target_img_h = (canvas_h // 16) * 16                                          # 720
    gen_size = (target_img_w, target_img_h)

    bg = _generate_background(script, gen_size)
    if bg is None:
        return None

    # Ghép bố cục split: Chữ trái, gradient, ảnh phải
    canvas, text_w = _compose_split(bg, (canvas_w, canvas_h), text_ratio, grad_ratio)
    final_img = _draw_split_title(canvas, script, text_w)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_img.save(out_path)
    log.info("Thumbnail AI bố cục split (trái/phải) đã tạo: %s", out_path)
    return out_path
