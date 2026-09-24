"""Thumbnail phong cách Kurzgesagt: AI sinh NỀN minh họa + PIL overlay chữ Việt.

Nền do Gemini (google-genai) sinh, KHÔNG chứa chữ (model viết chữ Việt hay lỗi).
Chữ tiêu đề được vẽ bằng PIL với font BeVietnamPro-Bold: trắng, viền đen dày,
to bản -> nét, đúng dấu, giống thumbnail Kurzgesagt.

Không có key / lỗi API -> trả None để caller fallback về make_thumbnail cũ.
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

    try:
        from .llm import generate

        out = generate(
            prompt=(
                "Video tiếng Việt sau đây cần một thumbnail minh họa.\n"
                f"Chủ đề: {context}\n\n"
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
    """Prompt tả NỀN Kurzgesagt bám CHỦ ĐỀ (dùng brief tiếng Anh từ LLM)."""
    brief = _visual_brief(script)
    return (
        f"Flat vector illustration in Kurzgesagt art style. Scene: {brief}. "
        "Bold saturated colors, smooth gradients, clean flat 2D shapes, soft glow, "
        "cinematic lighting, conceptual editorial illustration, digital wallpaper art. "
        "Composition keeps empty negative space on the left. 16:9 widescreen, scenery only."
    )


def _crop_to_16_9(img: Image.Image) -> Image.Image:
    """Cắt về 16:9, lệch xuống để bỏ chữ giả (mép trên) + logo giả (góc dưới)."""
    w, h = img.size
    # Phóng nhẹ + cắt biên để ăn hết viền màu FLUX hay để lại ở mép ảnh
    inset = int(min(w, h) * 0.03)
    img = img.crop((inset, inset, w - inset, h - inset))
    w, h = img.size
    target_h = int(w * 9 / 16)
    if target_h <= h:
        top = int((h - target_h) * 0.32)  # lệch xuống: cắt nhiều mép trên hơn
        return img.crop((0, top, w, top + target_h))
    target_w = int(h * 16 / 9)
    left = (w - target_w) // 2
    return img.crop((left, 0, left + target_w, h))


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
                return img.resize(size, Image.LANCZOS)
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
        img = client.text_to_image(_build_bg_prompt(script), model=model)
        return _crop_to_16_9(img.convert("RGB")).resize(size, Image.LANCZOS)
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
        return img.resize(size, Image.LANCZOS)
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


def _title_layout(img: Image.Image, title: str):
    """Tính font/dòng/vị trí tiêu đề (tách riêng để vẽ dải nền trước, chữ sau)."""
    W, H = img.size
    draw = ImageDraw.Draw(img)
    margin = int(W * 0.05)
    max_w = W - 2 * margin

    # Chọn cỡ chữ lớn nhất mà vẫn gói gọn <=3 dòng
    size = int(H * 0.20)
    while size > int(H * 0.09):
        font = _font(size)
        lines = _wrap(draw, title, font, max_w)
        if len(lines) <= 3:
            break
        size -= 6
    else:
        font = _font(size)
        lines = _wrap(draw, title, font, max_w)[:3]

    line_h = int(size * 1.12)
    total_h = line_h * len(lines)
    y = H - margin - total_h  # đặt cụm chữ ở NỬA DƯỚI cho dễ đọc
    return font, lines, y, line_h, size


def _title_band(size: tuple[int, int], top: int) -> Image.Image:
    """Dải gradient tối bán trong suốt sau tiêu đề: mờ dần lên trên để chữ luôn đọc rõ."""
    W, H = size
    band = Image.new("RGBA", size, (0, 0, 0, 0))
    top = max(0, min(top, H - 1))
    fade = max(1, int((H - top) * 0.45))  # đoạn chuyển mềm từ trong suốt -> tối
    max_a = 170
    od = ImageDraw.Draw(band)
    for j in range(top, H):
        a = int(max_a * (j - top) / fade) if (j - top) < fade else max_a
        od.line([(0, j), (W, j)], fill=(0, 0, 0, a))
    return band


def _draw_title(img: Image.Image, title: str) -> Image.Image:
    """Vẽ dải nền tối gradient + tiêu đề (trắng, viền đen dày) - style Kurzgesagt."""
    W, H = img.size
    margin = int(W * 0.05)
    font, lines, y, line_h, size = _title_layout(img, title)

    band_top = int(y - line_h * 0.35)
    img = Image.alpha_composite(img.convert("RGBA"), _title_band((W, H), band_top)).convert("RGB")

    draw = ImageDraw.Draw(img)
    stroke = max(6, size // 12)
    yy = y
    for line in lines:
        draw.text(
            (margin, yy), line, font=font, fill="#ffffff",
            stroke_width=stroke, stroke_fill="#000000",
        )
        yy += line_h
    return img


def make_ai_thumbnail(script: Script, out_path: Path) -> Path | None:
    """Sinh thumbnail Kurzgesagt (AI nền + chữ PIL). None nếu không sinh được nền."""
    size = (1280, 720)
    bg = _generate_background(script, size)
    if bg is None:
        return None

    bg = _draw_title(bg, script.title)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    bg.save(out_path)
    log.info("Thumbnail AI đã tạo: %s", out_path)
    return out_path
