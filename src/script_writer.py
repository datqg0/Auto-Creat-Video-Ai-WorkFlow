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
    mode = CONFIG.get("active_mode", "long")
    # ~150 từ/phút tiếng Việt khi đọc TTS. Cộng biên để chắc chắn đủ dài.
    approx_words = int(duration / 60 * 155)
    min_words = int(approx_words * 0.9)

    if mode == "short":
        return _build_short_prompt(topic, lang_name, duration, min_words, approx_words)

    minutes = max(duration // 60, 1)
    n_scenes_min = max(minutes * 2, 8)   # ~2 scene/phút -> đủ dài, không lê thê
    n_scenes_max = n_scenes_min + 4

    return f"""Viết kịch bản cho video YouTube dài ĐỦ {duration} giây (~{minutes} phút) về chủ đề:
"{topic}"

Ngôn ngữ: {lang_name}. Đây là yêu cầu độ dài BẮT BUỘC:
- Tổng lời đọc (cộng dồn tất cả narration) tối thiểu {min_words} từ, mục tiêu ~{approx_words} từ.
- Chia thành {n_scenes_min}-{n_scenes_max} scene, phủ ĐỦ 10 bước cấu trúc bên dưới.
- Mỗi scene narration 3-6 câu (khoảng 60-110 từ), KHÔNG viết narration 1 câu cụt.

CẤU TRÚC BẮT BUỘC (theo flow video giáo dục kiểu 3Blue1Brown, đi theo ĐÚNG thứ tự này,
mỗi bước là một hoặc vài scene liền mạch):
1. HOOK — câu hỏi/tình huống gây tò mò trong 10 giây đầu, khiến người xem muốn biết đáp án.
2. ĐẶT BÀI TOÁN — nêu rõ vấn đề cần giải quyết, vì sao nó quan trọng/khó.
3. TRỰC GIÁC / HÌNH DUNG — cho người xem "cảm" được vấn đề bằng hình ảnh, ẩn dụ, so sánh
   trước khi vào lý thuyết (nên dùng scene animation hoặc image_query mạnh ở đây).
4. Ý TƯỞNG — ý tưởng cốt lõi để giải quyết, giải thích "tại sao nó hoạt động".
5. VÍ DỤ TỪNG BƯỚC — đi qua một ví dụ cụ thể, từng bước một (nên dùng diagram/animation steps).
6. CÔNG THỨC / CODE — hình thức hóa bằng công thức hoặc code (scene code hoặc animation function).
7. DEMO — cho thấy nó chạy/áp dụng thực tế, kết quả cụ thể, con số.
8. BÀI TẬP / TỰ SUY NGHĨ — đặt 1 câu hỏi mở hoặc thử thách nhỏ cho người xem tự nghĩ.
9. TỔNG KẾT — chốt lại các ý chính, câu kết đáng nhớ + kêu gọi đăng ký kênh.
10. MỞ SANG VIDEO TIẾP THEO — gợi mở chủ đề liên quan sẽ nói ở video sau để giữ chân người xem.

Yêu cầu chiều sâu: giải thích "tại sao" và "cơ chế hoạt động" chứ không chỉ liệt kê "là gì";
có con số/ví dụ thực tế; nêu một hiểu lầm phổ biến nếu có.

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

QUAN TRỌNG về hình ảnh (video phải THẬT NHIỀU hình ảnh & animation, không được nhàm):
- Đa dạng visual_type: dùng ÍT NHẤT 5 loại khác nhau, KHÔNG để 2 scene bullets liên tiếp.
- BẮT BUỘC có TỐI THIỂU 3-5 scene "animation" rải đều trong video (hàm số, mạng neural,
  số liệu, thuật toán, quy trình) để video sinh động như 3Blue1Brown.
- MỌI scene (trừ code) đều PHẢI có "image_query": 2-5 từ khóa TIẾNG ANH mô tả ảnh minh họa nền
  cụ thể, sinh động (ví dụ "neural network brain glowing", "data center servers blue",
  "encryption padlock circuit", "quantum computer chip"). Không để trống.
- Ưu tiên hình ảnh trực quan hơn chữ: mỗi ý nên gắn với 1 hình ảnh hoặc animation minh họa.

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
- Scene đầu là HOOK (visual_type "title"), scene gần cuối là TỔNG KẾT ("quote"),
  scene cuối cùng là MỞ SANG VIDEO TIẾP THEO (gợi mở + call-to-action đăng ký).
- Bám sát 10 bước cấu trúc theo đúng thứ tự; heading mỗi scene nên phản ánh bước đang ở.
- Với "chart", số liệu hợp lý, labels/values cùng độ dài.
- Chỉ trả JSON, không markdown, không ```."""


def _build_short_prompt(
    topic: str, lang_name: str, duration: int, min_words: int, approx_words: int
) -> str:
    """Prompt cho YouTube Short: dọc 9:16, dưới 60s, nhịp nhanh, hook cực mạnh."""
    return f"""Viết kịch bản cho một YouTube SHORT (video DỌC 9:16, DƯỚI {duration} giây) về chủ đề:
"{topic}"

Ngôn ngữ: {lang_name}. Yêu cầu BẮT BUỘC cho định dạng Short:
- Tổng lời đọc chỉ khoảng {min_words}-{approx_words} từ (RẤT NGẮN, đọc trong <{duration}s). TUYỆT ĐỐI không dài hơn.
- Chia thành 4-6 scene ngắn, mỗi scene narration 1-2 câu (10-25 từ), nhịp nhanh, dứt khoát.
- Bố cục DỌC: chữ to, ít chữ mỗi màn hình, dễ đọc trên điện thoại.

CẤU TRÚC SHORT (rút gọn từ flow giáo dục, giữ nhịp nhanh):
1. HOOK cực mạnh trong 2 giây đầu — một câu hỏi sốc hoặc con số gây tò mò.
2. VẤN ĐỀ — nêu nhanh điều bất ngờ/khó hiểu.
3. GIẢI THÍCH TRỰC QUAN — 1-2 scene cốt lõi, dùng animation hoặc ảnh mạnh để "aha".
4. ĐIỂM CHỐT — insight/con số đáng nhớ nhất.
5. CALL-TO-ACTION — "Theo dõi để xem phần tiếp theo" hoặc câu hỏi mở kéo comment.

Yêu cầu hình ảnh cho Short:
- Mỗi scene PHẢI có "image_query" 2-5 từ khóa TIẾNG ANH, ảnh nổi bật, tương phản cao.
- Nên có 1-2 scene "animation" (counter con số, function, hoặc steps) để bắt mắt.
- Ưu tiên visual_type: "title", "quote", "animation"; hạn chế bullets dài.

Trả về DUY NHẤT một object JSON theo schema:
{{
  "title": "tiêu đề Short giật tít, có emoji hoặc con số, dưới 60 ký tự",
  "description": "1-2 câu + hashtag (#Shorts và 3-4 hashtag chủ đề) ở cuối",
  "tags": ["shorts", "tag2", "..."],
  "scenes": [
    {{
      "narration": "lời đọc ngắn, dứt khoát",
      "visual_type": "title",
      "heading": "chữ to hiển thị",
      "bullets": [],
      "chart": null,
      "code_language": "python",
      "algorithm": "",
      "image_query": "từ khóa ảnh tiếng Anh",
      "animation": null
    }}
  ]
}}

Lưu ý:
- Scene đầu = HOOK, scene cuối = CALL-TO-ACTION.
- Tổng lời đọc phải NGẮN để lọt dưới {duration} giây. Ưu tiên súc tích hơn đầy đủ.
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
