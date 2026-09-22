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
    minutes = max(duration // 60, 1)
    # ~150 từ/phút tiếng Việt khi đọc TTS. Cộng biên để chắc chắn đủ dài.
    approx_words = int(duration / 60 * 155)
    min_words = int(approx_words * 0.9)
    n_scenes_min = max(minutes * 2, 8)   # ~2 scene/phút -> đủ dài, không lê thê
    n_scenes_max = n_scenes_min + 4

    return f"""Viết kịch bản cho video YouTube dài ĐỦ {duration} giây (~{minutes} phút) về chủ đề:
"{topic}"

Ngôn ngữ: {lang_name}. Đây là yêu cầu độ dài BẮT BUỘC:
- Tổng lời đọc (cộng dồn tất cả narration) tối thiểu {min_words} từ, mục tiêu ~{approx_words} từ.
- Chia thành {n_scenes_min}-{n_scenes_max} scene.
- Mỗi scene narration 3-6 câu (khoảng 60-110 từ), KHÔNG viết narration 1 câu cụt.

Chiều sâu nội dung (để video có giá trị, không hời hợt):
- Mở đầu bằng một câu hỏi/tình huống gây tò mò (hook) trong 10 giây đầu.
- Giải thích "tại sao" và "cơ chế hoạt động", không chỉ liệt kê "là gì".
- Có ví dụ thực tế cụ thể, con số, hoặc so sánh dễ hình dung.
- Đề cập ứng dụng thực tế và một hiểu lầm phổ biến nếu có.
- Kết bằng tổng kết + câu chốt đáng nhớ và lời kêu gọi đăng ký kênh.

Mỗi scene có một "visual_type", chọn loại phù hợp nội dung:
- "title": scene mở đầu, chỉ có heading lớn.
- "bullets": liệt kê ý (2-4 gạch đầu dòng ngắn gọn).
- "chart": biểu đồ dữ liệu. Cần "chart": {{"kind": "bar|line|pie", "labels": [...], "values": [...]}}.
- "code": đoạn code minh họa. Đặt các dòng code vào "bullets", set "code_language".
- "algorithm": mô phỏng thuật toán. Set "algorithm" 1 trong: "sorting", "search", "graph", "neural_network".
- "diagram": sơ đồ luồng/quy trình. Đặt 3-5 bước ngắn vào "bullets" (mỗi bullet 1 bước).
- "quote": một câu chốt/ấn tượng, đặt câu đó vào bullets[0].
- "animation": clip ĐỘNG tự vẽ (giống 3Blue1Brown), rất bắt mắt. Set "animation" là 1 object:
  * đồ thị hàm số: {{"preset": "function", "expr": "sin(x)", "x_range": [-6.28, 6.28], "y_range": [-1.4, 1.4]}}
    (expr chỉ dùng x và các hàm sin,cos,tan,exp,log,sqrt,abs,tanh và +-*/^, hằng pi,e)
  * mạng neural: {{"preset": "neural_net", "layers": [3, 5, 4, 2]}}
  * biểu đồ cột động: {{"preset": "bar_chart", "values": [3,7,5,9], "labels": ["A","B","C","D"]}}
  * sắp xếp: {{"preset": "sorting", "data": [5,2,8,1,9,3]}}
  * con số đếm lên: {{"preset": "counter", "to_value": 1000000, "unit": "người dùng"}}
  * quy trình từng bước: {{"preset": "steps", "steps": ["Bước 1", "Bước 2", "Bước 3"]}}

QUAN TRỌNG về hình ảnh:
- Đa dạng visual_type: dùng ÍT NHẤT 4 loại khác nhau, tránh nhiều scene bullets liên tiếp.
- Nên có 1-3 scene "animation" khi nội dung phù hợp (hàm số, mạng neural, số liệu, thuật toán, quy trình)
  để video sinh động như 3Blue1Brown.
- Mỗi scene THÊM trường "image_query": 2-4 từ khóa TIẾNG ANH mô tả ảnh minh họa nền phù hợp
  (ví dụ "neural network brain", "server data center", "encryption lock"). Để "" nếu là code/chart.

Trả về DUY NHẤT một object JSON theo schema:
{{
  "title": "tiêu đề video hấp dẫn, có yếu tố tò mò, dưới 70 ký tự",
  "description": "mô tả 3-4 câu cho YouTube, có hashtag ở cuối",
  "tags": ["tag1", "tag2", "..."],
  "scenes": [
    {{
      "narration": "lời đọc tự nhiên, nhiều câu, kể chuyện lôi cuốn",
      "visual_type": "bullets",
      "heading": "tiêu đề ngắn hiển thị trên màn hình",
      "bullets": ["ý 1", "ý 2"],
      "chart": null,
      "code_language": "python",
      "algorithm": "",
      "image_query": "từ khóa ảnh tiếng Anh",
      "animation": null
    }}
  ]
}}

Lưu ý:
- narration phải liền mạch, kể chuyện, KHÔNG đọc gạch đầu dòng, KHÔNG quá ngắn.
- Scene đầu "title", scene cuối "quote" hoặc call-to-action.
- Với "chart", số liệu hợp lý, labels/values cùng độ dài.
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
