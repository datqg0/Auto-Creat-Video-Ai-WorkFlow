"""LLM viết kịch bản video chia theo scene, mỗi scene kèm visual spec."""
from __future__ import annotations

import json
import logging
import re

from .config import CONFIG
from .llm import generate
from .models import Scene, Script

log = logging.getLogger(__name__)

_SYSTEM = (
    "Bạn là biên kịch cho kênh YouTube giải thích công nghệ bằng hình ảnh trực quan "
    "(giống 3Blue1Brown, Kurzgesagt). Bạn luôn trả về JSON hợp lệ, không kèm giải thích."
)


def _build_prompt(topic: str) -> str:
    lang = CONFIG.get("language", "vi")
    lang_name = "tiếng Việt" if lang == "vi" else "English"
    duration = int(CONFIG.get("target_duration_seconds", 300))
    # ~150 từ/phút tiếng Việt khi đọc TTS
    approx_words = duration // 60 * 150

    return f"""Viết kịch bản cho video YouTube dài khoảng {duration} giây ({duration // 60} phút) về chủ đề:
"{topic}"

Ngôn ngữ: {lang_name}. Tổng lời đọc khoảng {approx_words} từ, chia thành 6-9 scene.

Mỗi scene có một "visual_type" trong các loại sau, chọn loại phù hợp nội dung:
- "title": scene mở đầu, chỉ có heading lớn.
- "bullets": liệt kê ý (2-4 gạch đầu dòng ngắn gọn).
- "chart": biểu đồ dữ liệu. Cần trường "chart": {{"kind": "bar|line|pie", "labels": [...], "values": [...]}}.
- "code": đoạn code minh họa. Đặt các dòng code vào "bullets", set "code_language".
- "algorithm": mô phỏng thuật toán. Set "algorithm" là 1 trong: "sorting", "search", "graph", "neural_network".
- "quote": một câu chốt/ấn tượng, đặt câu đó vào bullets[0].

Trả về DUY NHẤT một object JSON theo schema:
{{
  "title": "tiêu đề video hấp dẫn, có yếu tố tò mò, dưới 70 ký tự",
  "description": "mô tả 2-3 câu cho YouTube",
  "tags": ["tag1", "tag2", "..."],
  "scenes": [
    {{
      "narration": "lời đọc tự nhiên cho scene",
      "visual_type": "bullets",
      "heading": "tiêu đề ngắn hiển thị trên màn hình",
      "bullets": ["ý 1", "ý 2"],
      "chart": null,
      "code_language": "python",
      "algorithm": ""
    }}
  ]
}}

Lưu ý:
- narration phải liền mạch, kể chuyện lôi cuốn, KHÔNG đọc gạch đầu dòng.
- Scene đầu là "title", scene cuối nên là "quote" hoặc call-to-action.
- Với "chart", số liệu phải hợp lý và labels/values cùng độ dài.
- Chỉ trả JSON, không markdown, không ```."""


def _extract_json(text: str) -> dict:
    # Bỏ code fence nếu có
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    # Lấy object JSON đầu tiên
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Không tìm thấy JSON trong output LLM")
    return json.loads(text[start : end + 1])


def write_script(topic: str) -> Script:
    raw = generate(_build_prompt(topic), system=_SYSTEM)
    data = _extract_json(raw)

    scenes = [Scene(**s) for s in data.get("scenes", [])]
    if not scenes:
        raise ValueError("Kịch bản không có scene nào")

    script = Script(
        topic=topic,
        title=data.get("title", topic)[:100],
        description=data.get("description", ""),
        tags=data.get("tags", []),
        scenes=scenes,
    )
    log.info("Kịch bản '%s' có %d scene", script.title, len(scenes))
    return script
