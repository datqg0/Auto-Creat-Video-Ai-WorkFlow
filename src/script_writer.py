"""LLM viết kịch bản video chia theo scene, mỗi scene kèm visual spec."""
from __future__ import annotations

import json
import logging
import math
import re

from .config import CONFIG
from .llm import generate
from .models import Exercise, Scene, Script

log = logging.getLogger(__name__)

_SYSTEM = (
    "Bạn là biên kịch trưởng kiêm chuyên gia sư phạm công nghệ cho kênh YouTube giáo dục đỉnh cao "
    "(theo phong cách 3Blue1Brown, Veritasium, Kurzgesagt, Fireship). Triết lý sư phạm của bạn là "
    "CHIỀU SÂU BẢN CHẤT (DEPTH-FIRST PEDAGOGY): Thà giải thích 1 cơ chế đến tận cùng chân lý, "
    "còn hơn liệt kê 5 thuật ngữ chuyên ngành cưỡi ngựa xem hoa. "
    "BẮT BUỘC ÁP DỤNG 3 KỸ THUẬT SƯ PHẠM VÀNG: "
    "1. VẾT CHẠY VI MÔ (CONCRETE STEP-BY-STEP TRACE): Không nói lý thuyết trừu tượng; bắt buộc lấy 1 ca "
    "cụ thể (3 lệnh CPU cụ thể, mảng số cụ thể, gói tin mạng có số hiệu...) và truy vết từng chu kỳ/bước. "
    "2. ĐỒNG BỘ THỊ GIÁC & LỜI ĐỌC (DEICTIC NARRATION): Lời đọc phải 'trỏ tay vào hình vẽ' trên màn hình: "
    "chỉ rõ màu sắc, con trỏ, mũi tên, khối tín hiệu ('Hãy nhìn vào đường màu vàng này', 'Tại chu kỳ 3 con trỏ dừng lại vì...'). "
    "3. NGHỊCH LÝ & ĐÒN BẨY KỸ THUẬT (THE AHA HACK): Luôn chỉ rõ điểm bế tắc/xung đột của cách làm ngây thơ "
    "trước khi hé lộ cơ chế giải cứu thông minh và phân tích sự đánh đổi (Trade-off). "
    "YÊU CẦU TUYỆT ĐỐI VỀ ĐỘ CHÍNH XÁC (ZERO HALLUCINATION): Mọi độ phức tạp Big-O, chu kỳ xung nhịp, "
    "và số liệu kỹ thuật phải chuẩn xác 100% theo tiêu chuẩn khoa học máy tính quốc tế. "
    "Bạn luôn trả về JSON hợp lệ, không kèm giải thích ngoài."
)


def _as_question(title: str) -> str:
    """Đảm bảo tiêu đề ở dạng câu hỏi (kết thúc bằng '?')."""
    t = (title or "").strip().rstrip(".!…")
    if not t:
        return t
    if t.endswith("?"):
        return t
    return t + "?"


def _sentence_case(title: str) -> str:
    """Chỉ viết hoa chữ cái đầu câu, giữ nguyên phần còn lại (giữ acronym AI/GPU).

    Không dùng Title Case (Viết Hoa Mỗi Từ): với tiếng Việt trông như spam, giảm CTR.
    """
    t = (title or "").strip()
    return t[:1].upper() + t[1:] if t else t


def _build_prompt(topic: str, research_context: str = "", series_context: dict | None = None) -> str:
    lang = CONFIG.get("language", "vi")
    lang_name = "tiếng Việt" if lang == "vi" else "English"
    duration = int(CONFIG.get("target_duration_seconds", 300))
    mode = CONFIG.get("active_mode", "long")
    # Tốc độ đọc giáo dục thư thái: ~130 từ/phút tiếng Việt (kèm khoảng nghỉ giữa các câu để người xem kịp ngấm).
    approx_words = int(duration / 60 * 130)
    min_words = int(approx_words * 0.80) if mode == "short" else int(approx_words * 0.85)

    if mode == "short":
        return _build_short_prompt(topic, lang_name, duration, min_words, approx_words, series_context=series_context)

    minutes = max(duration // 60, 1)
    n_scenes_min = max(minutes * 2, 8)   # ~2 scene/phút -> nhịp điệu vừa vặn, không lê thê
    n_scenes_max = n_scenes_min + 4

    research_section = ""
    if research_context and research_context.strip():
        research_section = (
            f"\nTƯ LIỆU THỰC TẾ & SỐ LIỆU ĐÃ KIỂM CHỨNG (TỪ WEB RESEARCH AGENT):\n"
            f"{research_context.strip()}\n"
            f"(Hãy sử dụng các dữ kiện, độ phức tạp, và ví dụ thực chiến này trong các scene thích hợp)\n"
        )

    series_section = ""
    if series_context:
        s_name = series_context.get("series_name", "")
        ep_num = series_context.get("episode_num", 1)
        total_eps = series_context.get("total_episodes", 5)
        hook_prev = series_context.get("hook_from_previous", "")
        hook_next = series_context.get("hook_to_next", "")

        prev_clause = f"- Ở Scene 1 (Hook mở đầu): Nhắc lại gắn kết với tập trước: \"{hook_prev}\"" if (hook_prev and ep_num > 1) else ""
        next_clause = f"- Ở Scene cuối (Kết thúc): Gợi mở gây tò mò tập tiếp theo: \"{hook_next}\"" if (hook_next and ep_num < total_eps) else ""

        series_section = (
            f"\nNGỮ CẢNH CHUỖI VIDEO (SERIES):\n"
            f"- Video này là TẬP {ep_num}/{total_eps} thuộc chuỗi \"{s_name}\".\n"
            f"{prev_clause}\n"
            f"{next_clause}\n"
            f"- Tiêu đề kịch bản (title) NÊN có tiền tố '[Tập {ep_num}/{total_eps}]' để người xem dễ theo dõi.\n"
        )

    return f"""Viết kịch bản cho video YouTube dài ĐỦ {duration} giây (~{minutes} phút) về chủ đề:
"{topic}"
{research_section}{series_section}
Ngôn ngữ: {lang_name}. YÊU CẦU ĐỘ DÀI & PHÂN BỔ THỜI LƯỢNG (BẮT BUỘC):
- Tổng lời đọc (cộng dồn tất cả narration) tối thiểu {min_words} từ, mục tiêu ~{approx_words} từ (nhịp đọc thư thái ~130 từ/phút).
- Chia thành {n_scenes_min}-{n_scenes_max} scene theo CẤU TRÚC 5 PHA CHIỀU SÂU.
- ĐẶC BIỆT: PHA 3 (MỔ XẺ VI MÔ CƠ CHẾ KỸ THUẬT & TRACE TỪNG BƯỚC) BẮT BUỘC CHIẾM ĐỦ 50% TỔNG SỐ SCENE VÀ THỜI LƯỢNG LỜI ĐỌC!
- Mỗi scene narration 3-6 câu (~50-90 từ), câu ngắn gọn (8-16 từ), ngắt nhịp tự nhiên để người nghe kịp ngấm.

TRIẾT LÝ SƯ PHẠM CHIỀU SÂU (DEPTH-FIRST PEDAGOGY — CHỐNG CƯỠI NGỰA XEM HOA):

1. THÀ GIẢI THÍCH 1 CƠ CHẾ ĐẾN CÙNG, KHÔNG ĐIỂM DANH DÀN TRẢI:
   - TUYỆT ĐỐI KHÔNG điểm danh 4-5 khái niệm lướt qua bề mặt.
   - CHỈ TẬP TRUNG vào 1 bài toán trung tâm + 1 cơ chế kỹ thuật then chốt nhất và bóc tách nó đến tận tầng vi mô (clock cycle / byte / register / packet / memory).

2. BẮT BUỘC CÓ "VẾT CHẠY VI MÔ" (CONCRETE STEP-BY-STEP TRACE):
   - BẮT BUỘC chọn 1 CA THỰC THI CỤ THỂ XUYÊN SUỐT với dữ liệu/số liệu thực:
     * CPU/Phần cứng: 3 lệnh hợp ngữ cụ thể (Lệnh 1: R1 = R2 + R3; Lệnh 2: R4 = R1 + 5; Lệnh 3: Branch...), mổ xẻ từng chu kỳ xung nhịp (Clock Cycle 1, 2, 3, 4, 5).
     * Thuật toán: Mảng số cụ thể [5, 2, 8, 1], lần theo từng bước so sánh, hoán đổi và con trỏ.
     * Mạng/Hệ thống: Client gửi gói tin SYN seq=100, Server phản hồi ACK=101, mô phỏng tình huống mất gói tin ở bước nào.
   - Phân tích rõ 3 chặng của ca cụ thể này:
     * Chặng A (Bình thường): Dữ liệu di chuyển qua các trạm.
     * Chặng B (Xung đột / Nút thắt): Điểm nghẽn xuất hiện! Lệnh 2 cần R1 nhưng Lệnh 1 chưa kịp ghi vào thanh ghi -> CPU bị khựng (Stall / Bubble) mất 2 chu kỳ thế nào!
     * Chặng C (Cú hack giải cứu - The Aha Mechanism): Mạch Bypass / Forwarding truyền tắt dữ liệu ngay từ đầu ra ALU sang đầu vào ALU tiếp theo, xóa sạch 2 chu kỳ khựng!

3. KỸ THUẬT "DEICTIC NARRATION" (LỜI ĐỌC TRỎ VÀO HÌNH VẼ — BẮT BUỘC):
   - Mọi scene animation, lời đọc PHẢI trực tiếp hướng dẫn mắt người xem:
     * ĐÚNG: "Hãy nhìn vào đường dây màu vàng này: ngay khi ALU hoàn thành phép tính, tín hiệu được chuyển tắt trực tiếp...", "Tại chu kỳ 3, ô màu hồng này khựng lại vì dữ liệu chưa sẵn sàng...", "Mũi tên màu xanh neon này là đường bypass truyền tắt kết quả..."
     * SAI: "CPU thực hiện các lệnh liên tục với tốc độ cao." (quá chung chung, không ăn nhập hình vẽ).
   - visual_prompt vẽ cái gì thì narration phải gọi tên đúng màu sắc, vị trí, hoặc trạng thái đó!

CẤU TRÚC 5 PHA BẮT BUỘC:

PHA 1: HOOK & NGHỊCH LÝ BẾ TẮC (~15% thời lượng, 1-2 scene)
- Scene 1 BẮT BUỘC visual_type: "animation" do AI code vẽ ngay từ giây thứ 0.
- Câu đầu tiên là một cú sốc / nghịch lý phản trực giác. Nêu bài toán: Tại sao cách làm tự nhiên lại chậm chạp, tắc nghẽn hoặc thất bại? TUYỆT ĐỐI không mở đầu bằng "Xin chào", "Hôm nay chúng ta". Vào thẳng vấn đề!

PHA 2: MENTAL MODEL & ẨN DỤ TRỰC GIÁC ĐỜI THƯỜNG (~15% thời lượng, 1-2 scene)
- 1 ẩn dụ đời thực cực đắt (ví dụ: máy giặt sấy gối đầu, thủ thư đánh số ngăn kéo, cao tốc phân luồng...).
- Giúp người xem "cảm nhận được bằng trực giác" cách giải quyết trước khi nghe thuật ngữ.

PHA 3: MỔ XẺ VI MÔ CƠ CHẾ KỸ THUẬT & TRACE TỪNG BƯỚC (50% THỜI LƯỢNG — TRỌNG TÂM CHIẾM NỬA VIDEO, 4-6 scene liên tiếp)
- Toàn bộ thời lượng này dùng để TRACE CA CỤ THỂ đã chọn ở trên qua từng chu kỳ/bước.
- Đi qua tuần tự: Trạng thái bình thường -> Xung đột phát sinh (Hazard/Stall/Drop) -> Cú hack kỹ thuật tháo gỡ (Bypass/Sliding Window/Tree Rebalance) -> Kết quả định lượng (giảm từ 12 chu kỳ xuống 5 chu kỳ).
- 100% các scene trong Pha 3 NÊN là animation code ("pycode" hoặc "manim"), áp dụng triệt để DEICTIC NARRATION.

PHA 4: ỨNG DỤNG THỰC CHIẾN & ĐÁNH ĐỔI (TRADE-OFFS) (~15% thời lượng, 1-2 scene)
- Phân tích cái giá phải trả: Tốn thêm bóng bán dẫn, ngốn bộ nhớ, tăng độ phức tạp mạch điện, hay rủi ro bảo mật (như Spectre)?
- Các hệ thống lớn (Linux kernel, Intel Core, Apple Silicon, V8...) xử lý đánh đổi này ra sao.

PHA 5: BẢN CHẤT TRONG 1 CÂU & KẾ THỪA TẬP SAU (~5% thời lượng, 1 scene)
- Đúc kết insight đắt giá nhất trong 1 câu chốt.
- Đặt câu hỏi kích thích tư duy hoặc gợi mở tập tiếp theo của chuỗi video.

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
  * TỰ THIẾT KẾ animation riêng (giống 3Blue1Brown, ƯU TIÊN DÙNG để minh họa
    phong phú hơn): {{"preset": "custom", "objects": [...], "timeline": [...]}}    - "objects": danh sách phần tử, mỗi phần tử có "id" (duy nhất) + "type":
      · text  : {{"id":"t1","type":"text","text":"E=mc^2","x":"0.5W","y":160,"size":72,"color":"accent","bold":true,"anchor":"mm"}}
      · axes  : {{"id":"ax","type":"axes","x0":"0.14W","y0":320,"x1":"0.86W","y1":"0.85H","x_range":[-6.28,6.28],"y_range":[-1.4,1.4]}}
      · graph : {{"id":"g","type":"graph","axes":"ax","expr":"sin(x)","color":"c1","glow":0.6}}  (expr chỉ dùng x, sin,cos,tan,exp,log,sqrt,abs,tanh, +-*/^, pi,e; "glow" 0..1 tạo hào quang neon)
      · parametric: {{"id":"p","type":"parametric","axes":"ax","expr_x":"cos(t)","expr_y":"sin(t)","t_range":[0,6.28],"color":"c2","glow":0.6}}  (đường cong tham số theo biến t; toán an toàn giống graph; "glow" neon tuỳ chọn)
      · formula: {{"id":"f","type":"formula","text":"\\\\frac{{d}}{{dx}}e^x = e^x","x":"0.5W","y":180,"size":72,"color":"accent"}}  (công thức đẹp KIỂU LaTeX; dùng cho phương trình toán học)
      · rect  : {{"id":"b","type":"rect","x0":"0.3W","y0":500,"x1":"0.7W","y1":620,"fill":"panel","outline":"accent","radius":16}}
      · circle: {{"id":"c","type":"circle","x":"0.5W","y":"0.5H","radius":80,"outline":"accent"}}
      · polygon: {{"id":"pg","type":"polygon","points":[["0.4W",400],["0.6W",400],["0.5W",600]],"outline":"accent","fill":"panel","glow":0.5}}  (đa giác đóng >=3 đỉnh; dùng làm target cho "vmorph")
      · line/arrow: {{"id":"ar","type":"arrow","x1":"0.5W","y1":640,"x2":"0.5W","y2":760,"color":"muted"}}
      · dot   : {{"id":"d","type":"dot","x":"0.5W","y":"0.5H","radius":12,"color":"c3"}}
      · neural_net: {{"id":"nn","type":"neural_net","layers":[3,5,4,2],"x0":"0.2W","y0":320,"x1":"0.8W","y1":"0.85H"}}
      · bar_chart: {{"id":"bc","type":"bar_chart","values":[3,7,5],"labels":["A","B","C"],"x0":"0.16W","y0":340,"x1":"0.84W","y1":"0.85H"}}
    - Toạ độ: số px (khung 1920x1080), hoặc chuỗi "0.5W"/"0.85H" (phần trăm khung).
    - Màu: hex "#3fb950", tên theme (accent/text/muted/panel/grid), hoặc "c0".."c6".
    - "timeline": danh sách bước, mỗi bước LÀ MỘT trong:
      · {{"play": [{{"anim":"write","target":"t1","run_time":1.0}}, ...]}}  (chạy song song các anim trong list)
      · {{"wait": 0.5}}
      anim hợp lệ: "fade_in"(shift), "fade_out", "write"(chữ), "draw"(line/arrow/graph/parametric/bar_chart),
      "grow", "move"(dx,dy), "count_up"(from,to,fmt), "pulse"(amount,cycles), "signal"(neural_net),
      "camera"(pan chuyển động nhẹ góc nhìn: {{"anim":"camera","cx":"0.5W","cy":"0.5H","run_time":1.2}} — KHÔNG dùng zoom để tránh cắt mất nội dung),
      "transform"/"morph"(biến hình A->B: {{"anim":"transform","target":"c","to":"b","run_time":1.0}} — cần "to" là id đích),
      "move_along"(chạy 1 dot dọc theo graph/parametric: {{"anim":"move_along","target":"d","path":"p","trace":true,"run_time":2.0}} — "path" là id graph/parametric, "trace":true vẽ dần nét ngay dưới điểm chạy),
      "vmorph"(biến hình THỰC theo đỉnh: {{"anim":"vmorph","target":"pg","from":"c","to":"b","run_time":1.5}} — "target" phải là polygon, "from"/"to" là id circle/rect/polygon; mượt hơn "transform").
  * TỰ VIẾT CODE Python (matplotlib/manim) để vẽ animation phức tạp:
    {{"preset": "pycode", "code": "..."}} hoặc {{"preset": "manim", "code": "..."}}

QUAN TRỌNG VỀ NHỊP VĂN & GIỌNG ĐỌC:
- NHỊP ĐỌC ĐÀM THOẠI & CÂU NGẮN (8-16 TỪ): Viết câu ngắn gọn, gãy gọn, giàu tính đàm thoại.
- LỜI ĐỌC TỰ NHIÊN, VĂN BẢN THUẦN: Dùng ngôn ngữ bình dân, gợi hình, so sánh trực quan.
  TUYỆT ĐỐI KHÔNG dùng dấu sao (* hoặc **) hoặc gạch dưới để bôi đậm từ.

QUAN TRỌNG VỀ HÌNH ẢNH & VISUAL PROMPT:
- BẮT BUỘC có ÍT NHẤT 70% số scene có visual_type: "animation" do code AI viết (preset "pycode" hoặc "manim").
- Scene 1 (HOOK) BẮT BUỘC có visual_type: "animation" do code AI vẽ ngay từ giây đầu tiên.
- TIÊU CHUẨN VÀNG CHO "visual_prompt" (ĐỒNG BỘ VỚI DEICTIC NARRATION):
  * BẮT BUỘC mô tả 3 yếu tố:
    1. DỮ LIỆU CỤ THỂ: 3 lệnh CPU, mảng 4 số, 3 node mạng...
    2. CHUYỂN ĐỘNG TỪNG BƯỚC & MÀU SẮC: Ô nào sáng màu Vàng, đường dẫn nào đổi màu Xanh neon, con trỏ dừng ở đâu.
    3. NHỊP ĐIỆU: Dừng 0.6-0.8s ở mỗi bước để người xem kịp ngấm.
- MỌI scene (trừ code) đều PHẢI có "image_query": 2-5 từ khóa TIẾNG ANH mô tả ảnh minh họa nền.
- "video_query": với các scene hợp với cảnh quay thực tế (data center, vi mạch...), thêm 2-4 từ khóa TIẾNG ANH tải footage video.

Trả về DUY NHẤT một object JSON theo schema:
{{
  "title": "tiêu đề gây tò mò mạnh, dưới 70 ký tự (xem quy tắc tiêu đề bên dưới)",
  "description": "mô tả 3-4 câu cho YouTube, có hashtag ở cuối",
  "tags": ["tag1", "tag2", "..."],
  "scenes": [
    {{
      "narration": "lời đọc tự nhiên, câu ngắn dưới 18 từ có DEICTIC NARRATION chỉ vào hình vẽ (TUYỆT ĐỐI không dùng dấu * hoặc **)",
      "visual_type": "animation",
      "heading": "tiêu đề ngắn hiển thị trên màn hình",
      "visual_prompt": "prompt chi tiết về dữ liệu, màu sắc và chuyển động từng bước để AI viết code Python/Manim vẽ khớp lời đọc",
      "bullets": [],
      "chart": null,
      "code_language": "python",
      "algorithm": "",
      "image_query": "từ khóa ảnh tiếng Anh",
      "video_query": "từ khóa footage video tiếng Anh (hoặc để trống)",
      "animation": {{"preset": "pycode"}}
    }}
  ],
  "exercises": [
    {{
      "question": "một bài toán/tình huống THỰC TẾ với số liệu cụ thể để người xem tự giải",
      "hint": "gợi ý ngắn hướng giải (không bắt buộc)",
      "answer": "đáp án hoặc hướng làm ngắn gọn (không bắt buộc)"
    }}
  ]
}}

BÀI TẬP VÍ DỤ (BẮT BUỘC): tạo mảng "exercises" gồm ĐÚNG 1 bài toán/tình huống THỰC TẾ tiêu biểu nhất ở cuối video.

Lưu ý tiêu đề:
- "title" phải CHỌN MỘT trong các dạng: Câu hỏi gây sốc, Con số + lợi ích rõ ràng, Khoảng cách tò mò, Cổ phần khẩn cấp, Phản trực giác.
  TUYỆT ĐỐI KHÔNG dùng: "Giới thiệu về...", "Tìm hiểu...", "Hướng dẫn...".
- Chỉ trả JSON, không markdown, không ```."""


def _build_short_prompt(
    topic: str,
    lang_name: str,
    duration: int,
    min_words: int,
    approx_words: int,
    series_context: dict | None = None,
) -> str:
    """Prompt cho YouTube Short: dọc 9:16, tối đa 3 phút (180s), tập trung 1 ca vi mô có Aha moment."""
    n_scenes_min = max(int(duration / 25), 4)
    n_scenes_max = max(int(duration / 15), 6)

    series_clause = ""
    if series_context:
        s_name = series_context.get("series_name", "")
        ep_num = series_context.get("episode_num", 1)
        total_eps = series_context.get("total_episodes", 5)
        series_clause = f"\nThuộc chuỗi: '{s_name}' [Tập {ep_num}/{total_eps}].\n"

    return f"""Viết kịch bản cho một YouTube SHORT (video DỌC 9:16, thời lượng mục tiêu ~{duration} giây, TUYỆT ĐỐI không vượt quá 180 giây / 3 phút) về chủ đề:
"{topic}"
{series_clause}
Ngôn ngữ: {lang_name}. TRIẾT LÝ SƯ PHẠM SHORT (CHIỀU SÂU & 1 SINGLE AHA MOMENT):
- Tổng lời đọc khoảng {min_words}-{approx_words} từ (nhịp đọc thư thái, rõ ràng, TUYỆT ĐỐI không vượt quá 180s).
- Chia thành {n_scenes_min}-{n_scenes_max} scene ngắn gọn.
- Bố cục DỌC 9:16: chữ TO, RẤT ÍT chữ mỗi màn hình để không tràn khung trên điện thoại.
- KHÔNG CƯỠI NGỰA XEM HOA: Chọn ĐÚNG 1 VÍ DỤ VI MÔ CỤ THỂ (1 bài toán nghẽn -> cách giải quyết thông minh).
- DEICTIC NARRATION: Lời thoại chỉ thẳng vào hình vẽ trên màn hình ("Nhìn vào con số màu đỏ này...", "Chính đường dây màu xanh này...").

CẤU TRÚC SHORT 4 PHA:
1. HOOK CỰC MẠNH (Scene 1 — 0-5s): Một nghịch lý phản trực giác hoặc câu hỏi gây sốc khiến người xem dừng lướt. BẮT BUỘC visual_type: "animation".
2. ẨN DỤ TỨC THÌ (Scene 2): 1 so sánh đời thực cực nhanh để não bộ hình dung ngay.
3. TRACE VI MÔ & CÚ HACK (Scene 3-4, chiếm 60% thời lượng): Mổ xẻ 1 ví dụ cụ thể có xung đột và cách giải quyết thông minh bằng animation code.
4. INSIGHT CHỐT & KÊU GỌI (Scene cuối): Đúc kết 1 câu bản chất nhất + câu hỏi kích thích bình luận.

Trả về DUY NHẤT một object JSON theo schema:
{{
  "title": "tiêu đề là MỘT CÂU HỎI giật tít mà Short sẽ trả lời (kết thúc bằng ?), dưới 60 ký tự",
  "description": "1-2 câu + hashtag (#Shorts và 3-4 hashtag chủ đề) ở cuối",
  "tags": ["shorts", "tag2", "..."],
  "scenes": [
    {{
      "narration": "lời đọc ngắn gọn có deictic narration (TUYỆT ĐỐI không dùng dấu * hoặc **)",
      "visual_type": "animation",
      "heading": "chữ to hiển thị",
      "visual_prompt": "prompt mô tả animation dọc 9:16 có chuyển động rõ ràng, nhịp chậm rãi",
      "bullets": [],
      "chart": null,
      "code_language": "python",
      "algorithm": "",
      "image_query": "từ khóa ảnh tiếng Anh",
      "video_query": "từ khóa footage video tiếng Anh (hoặc để trống)",
      "animation": {{"preset": "pycode"}}
    }}
  ],
  "exercises": [
    {{
      "question": "một câu đố/tình huống tư duy nhanh",
      "hint": "",
      "answer": ""
    }}
  ]
}}
Chỉ trả JSON, không markdown, không ```."""


def _extract_json(text: str) -> dict:
    # Bỏ code fence nếu có
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    # Lấy object JSON đầu tiên
    start = text.find("{")
    end = text.rfind("}")
    if start == -1:
        raise ValueError("Không tìm thấy JSON trong output LLM")

    # Thử parse chuẩn trước nếu có đóng ngoặc đầy đủ
    if end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass

    # Fallback tự động sửa JSON (xử lý unescaped quote, trailing comma, hoặc JSON bị cắt cụt)
    try:
        import json_repair

        repaired = json_repair.repair_json(text[start:], return_objects=True)
        if isinstance(repaired, dict):
            return repaired
        if isinstance(repaired, list) and repaired and isinstance(repaired[0], dict):
            return repaired[0]
    except Exception as err:
        log.warning("json_repair thất bại: %s", err)

    if end != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("Không tìm thấy JSON hợp lệ trong output LLM")


def _count_words(scenes: list[Scene]) -> int:
    """Đếm tổng số từ lời đọc (narration) để kiểm tra kịch bản đủ dài chưa."""
    return sum(len((s.narration or "").split()) for s in scenes)


def _optimize_title(script: Script) -> str:
    """Lượt LLM riêng chuyên viết tiêu đề: sinh 5 phương án, tự chọn tốt nhất.

    Chạy SAU khi kịch bản đã xong nên biết chính xác nội dung, số liệu, insight
    chính của video → tiêu đề không bị chung chung.
    Thất bại thì trả về title gốc (LLM viết khi sinh kịch bản).
    """
    lang = CONFIG.get("language", "vi")
    lang_name = "tiếng Việt" if lang == "vi" else "English"

    hook = ""
    if script.scenes:
        hook = (script.scenes[0].narration or "")[:300]

    prompt = f"""Bạn là chuyên gia tiêu đề YouTube. Video về chủ đề: "{script.topic}".
Tiêu đề hiện tại: "{script.title}"
Câu mở đầu video (hook): "{hook}"

Viết 5 tiêu đề {lang_name} KHÁC NHAU theo ĐÚNG 5 dạng sau (mỗi dạng 1 tiêu đề):
1. CÂU HỎI GÂY SỐC: bắt đầu bằng câu hỏi có con số hoặc nghịch lý chưa ai ngờ đến.
   Ví dụ tốt: "Tại sao 99% lập trình viên hiểu sai cách RAM hoạt động?"
2. CON SỐ + LỢI ÍCH: dùng con số cụ thể, hứa hẹn rõ người xem được gì.
   Ví dụ tốt: "5 phút hiểu thuật toán tìm đường ngắn nhất mà Google Maps dùng hàng ngày"
3. KHOẢNG CÁCH TÒ MÒ: gợi ra điều bí ẩn mà chưa tiết lộ đáp án.
   Ví dụ tốt: "Thứ ẩn bên trong mọi ảnh JPEG mà bạn không thể nhìn thấy"
4. CỔ PHẦN CAO: nêu hậu quả/nguy hiểm nếu không biết, hoặc lợi ích lớn nếu biết.
   Ví dụ tốt: "Lỗ hổng này làm 500 triệu tài khoản bị hack — code của bạn có dính không?"
5. PHẢN TRỰC GIÁC: một tuyên bố nghe vô lý nhưng hóa ra đúng.
   Ví dụ tốt: "Thêm máy chủ đôi khi khiến hệ thống chậm hơn — đây là lý do"

Quy tắc:
- Mỗi tiêu đề ≤ 70 ký tự (đủ hiển thị trên điện thoại).
- Không dùng dấu ngoặc kép bên trong tiêu đề.
- Không bắt đầu bằng "Giới thiệu", "Tìm hiểu", "Hướng dẫn", "Trong video này".
- Bám vào NỘI DUNG THẬT của video (dùng con số, ví dụ, insight từ hook bên trên).

Sau đó chọn 1 tiêu đề tốt nhất theo tiêu chí: CTR cao nhất (người lạ bấm vào nhiều nhất),
có từ khóa tìm kiếm, và khớp nội dung video.

Chỉ trả về JSON:
{{"candidates": ["tiêu đề 1", "tiêu đề 2", "tiêu đề 3", "tiêu đề 4", "tiêu đề 5"], "best": "tiêu đề được chọn"}}"""

    try:
        raw = generate(prompt, system="Bạn là chuyên gia tối ưu tiêu đề YouTube. Chỉ trả JSON hợp lệ.", task="topic")
        data = _extract_json(raw)
        best = (data.get("best") or "").strip()
        candidates = data.get("candidates", [])
        log.info(
            "Tiêu đề tối ưu: %s (các phương án: %s)",
            best, " | ".join(str(c) for c in candidates[:5]),
        )
        return _sentence_case(best)[:100] if best else script.title
    except Exception as e:  # noqa: BLE001 - không làm hỏng cả pipeline
        log.warning("Tối ưu tiêu đề lỗi, giữ title gốc: %s", e)
        return script.title


def _clean_narration_markdown(text: str) -> str:
    """Xóa bỏ triệt để ký tự markdown (*, **, _, ~, `), emoji, bullet, URL để lời đọc sạch sẽ."""
    if not text:
        return ""
    # Bỏ link web URL nếu có lọt vào
    t = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Bỏ markdown links: [text](url) -> text
    t = re.sub(r"\[([^\]]+)\]\([^\)]*\)", r"\1", t)
    # Bỏ markdown bold/italic: ***text***, **text**, *text*, ___text___, __text__, _text_
    t = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", t)
    t = re.sub(r"_{1,3}(.*?)_{1,3}", r"\1", t)
    t = re.sub(r"~~(.*?)~~", r"\1", t)
    t = re.sub(r"`+([^`]+)`+", r"\1", t)
    # Bỏ emoji / unicode biểu tượng (🔥, 🚀, 👋, 💡...)
    t = re.sub(r"[\U00010000-\U0010ffff]", "", t)
    # Bỏ các ký hiệu gây đọc lạ: bullet •, hashtag #, sao *, gạch dưới _, ngã ~, backtick `, gạch đứng |, gạch chéo \, mũ ^, ngoặc nhọn, ngoặc vuông
    t = re.sub(r"[•#\*\_~`|\\^<>{}\[\]]", " ", t)
    # Bỏ gạch nối đơn độc (tránh đọc thành 'trừ' hoặc 'gạch')
    t = re.sub(r"(?:^|\s)[-\u2013\u2014]+(?:\s|$)", " ", t)
    # Chuẩn hóa khoảng trắng
    return re.sub(r"\s+", " ", t).strip()


def _parse_script(topic: str, raw: str) -> Script:
    data = _extract_json(raw)

    scenes: list[Scene] = []
    for s in data.get("scenes", []):
        if not isinstance(s, dict):
            continue
        if "narration" in s and isinstance(s["narration"], str):
            s["narration"] = _clean_narration_markdown(s["narration"])
        if "heading" in s and isinstance(s["heading"], str):
            s["heading"] = _clean_narration_markdown(s["heading"])
        if "bullets" in s and isinstance(s["bullets"], list):
            s["bullets"] = [_clean_narration_markdown(b) if isinstance(b, str) else b for b in s["bullets"]]
        try:
            scenes.append(Scene(**s))
        except Exception as e:  # noqa: BLE001 - bỏ qua scene lỗi, không giết cả run
            log.warning("Bỏ qua scene lỗi: %s", e)
    if not scenes:
        raise ValueError("Kịch bản không có scene nào")

    exercises: list[Exercise] = []
    for ex in data.get("exercises", []):
        try:
            if isinstance(ex, str):
                exercises.append(Exercise(question=ex))
            elif isinstance(ex, dict):
                exercises.append(Exercise(**ex))
        except Exception as e:  # noqa: BLE001 - bỏ qua bài lỗi
            log.warning("Bỏ qua bài toán lỗi: %s", e)

    return Script(
        topic=topic,
        title=_sentence_case(_as_question(data.get("title", topic)))[:100],
        description=data.get("description", ""),
        tags=data.get("tags", []),
        scenes=scenes,
        exercises=exercises,
    )


def _write_mega_script(topic: str, duration: int, research_context: str = "") -> Script:
    """Tạo kịch bản siêu dài (mega mode >= 10-30 phút) bằng phương pháp phân tầng (Curriculum Chapters).

    1. Sinh dàn ý mục lục 4-7 chương (Chapters/Modules).
    2. Viết chi tiết từng chương (mỗi chương 4-6 scene với >=70% animation code).
    3. Ghép nối thành 1 kịch bản Script hoàn chỉnh có chiều sâu cực đại.
    """
    lang = CONFIG.get("language", "vi")
    lang_name = "tiếng Việt" if lang == "vi" else "English"
    approx_words = int(duration / 60 * 130)
    minutes = max(duration // 60, 10)
    num_chapters = min(max(minutes // 3, 4), 7)  # 4-7 chương tùy độ dài

    log.info("Bắt đầu sinh kịch bản siêu dài (Mega Deep-Dive): %s (%d phút, %d chương)", topic, minutes, num_chapters)

    research_section = ""
    if research_context and research_context.strip():
        research_section = f"\nTƯ LIỆU THỰC TẾ & BẰNG CHỨNG (TỪ WEB RESEARCH):\n{research_context.strip()}\n"

    outline_prompt = f"""Bạn là đạo diễn và biên kịch trưởng cho kênh YouTube giải thích công nghệ chuyên sâu (kiểu 3Blue1Brown, Fireship, ByteByteGo).
Chúng ta đang sản xuất video MASTERCLASS SIÊU DÀI ĐẶC BIỆT (~{minutes} phút, mục tiêu ~{approx_words} từ lời đọc) về chủ đề:
"{topic}"
{research_section}
Hãy lập DÀN Ý MỤC LỤC CHI TIẾT gồm {num_chapters} chương lớn (Chapters) theo thứ tự sư phạm xuất sắc từ con số 0 đến làm chủ hoàn toàn:
1. Chương 1: Cú Móc & Nghịch lý Cốt lõi (Hook, Paradox & Why It Matters) - Scene 1 BẮT BUỘC là animation do code vẽ.
2. Chương 2: Mô hình Trực giác & Bức tranh Tổng quan (Mental Model & High-level Architecture).
3. Chương 3: Cơ chế Vận hành Từng bước (Step-by-step Technical Mechanism & Dataflow).
4. Chương 4: Đi sâu vào Bản chất Toán học / Thuật toán / Code (Under the Hood, Math & Algorithms).
5. Chương 5: Thử nghiệm Thực tế, Đột phá & So sánh Hiệu năng (Real-world Benchmarks & Implementation).
{f"6. Chương 6: Cạm bẫy Phổ biến & Ứng dụng Nâng cao (Edge cases & Production Pitfalls)." if num_chapters >= 6 else ""}
{f"7. Chương 7: Tương lai, Bài toán Mở & Tổng kết (Future Horizon & Next Big Challenge)." if num_chapters >= 7 else "6. Chương cuối: Tương lai, Bài toán Mở & Tổng kết."}

Trả về DUY NHẤT một object JSON theo schema:
{{
  "title": "Tiêu đề video cực kỳ hấp dẫn, gây tò mò, dưới 70 ký tự (kết thúc bằng ? hoặc khẳng định sốc)",
  "description": "Mô tả chuyên sâu 4-5 câu cho video Masterclass YouTube, kèm hashtag",
  "tags": ["tag1", "tag2", "tag3", "tag4", "tag5"],
  "chapters": [
    {{
      "chapter_number": 1,
      "chapter_title": "Tên chương ngắn gọn",
      "objective": "Mục tiêu cụ thể và các khái niệm cần mổ xẻ trong chương này",
      "target_scenes": 5
    }}
  ]
}}
Chỉ trả về JSON thuần, không kèm markdown hay lời dẫn."""

    raw_outline = generate(outline_prompt, system=_SYSTEM, task="script")
    data_outline = _extract_json(raw_outline)
    title = _sentence_case(_as_question(data_outline.get("title", topic)))[:100]
    description = data_outline.get("description", "")
    tags = data_outline.get("tags", [])
    chapters = data_outline.get("chapters", [])
    if not chapters:
        chapters = [
            {"chapter_number": 1, "chapter_title": "Khởi nguyên & Cú móc", "objective": "Đặt vấn đề và nghịch lý", "target_scenes": 5},
            {"chapter_number": 2, "chapter_title": "Mô hình Trực giác", "objective": "Bức tranh tổng quan", "target_scenes": 5},
            {"chapter_number": 3, "chapter_title": "Cơ chế Vận hành", "objective": "Chi tiết thuật toán", "target_scenes": 6},
            {"chapter_number": 4, "chapter_title": "Kiến trúc & Toán học", "objective": "Mổ xẻ tầng sâu", "target_scenes": 6},
            {"chapter_number": 5, "chapter_title": "Thực tế & Tổng kết", "objective": "Áp dụng thực tế và bài tập", "target_scenes": 5},
        ]

    all_scenes: list[Scene] = []
    total_words_target_per_chapter = approx_words // len(chapters)

    for ch_idx, ch in enumerate(chapters):
        ch_num = ch.get("chapter_number", ch_idx + 1)
        ch_title = ch.get("chapter_title", f"Chương {ch_num}")
        ch_obj = ch.get("objective", "")
        n_scenes = int(ch.get("target_scenes", 5))
        is_first_ch = (ch_idx == 0)

        log.info("Đang viết Chương %d/%d: %s...", ch_num, len(chapters), ch_title)

        ch_prompt = f"""Bạn đang viết nội dung chi tiết cho Chương {ch_num}: "{ch_title}"
Nằm trong video Masterclass: "{title}"
Mục tiêu chương này: {ch_obj}

Yêu cầu BẮT BUỘC cho chương này ({lang_name}):
- Tạo đúng {n_scenes} scene liền mạch, giải thích sâu sắc đến tận tầng vi mô (clock cycle / byte / register / packet / RAM), lời đọc tự nhiên (~{total_words_target_per_chapter} từ cho cả chương).
- VẾT CHẠY VI MÔ (CONCRETE TRACE): Lấy 1 ví dụ cụ thể với dữ liệu thực, lần theo từng bước/chu kỳ để chỉ rõ điểm nghẽn và cú hack tháo gỡ.
- DEICTIC NARRATION: Lời thoại trỏ trực tiếp vào hình vẽ trên màn hình ("Hãy nhìn vào đường tín hiệu màu vàng...", "Tại chu kỳ 3 ô màu hồng khựng lại vì...").
- Mỗi scene narration dài 3-6 câu (~60-100 từ), câu ngắn dưới 20 từ, TUYỆT ĐỐI không dùng dấu hoa thị/sao (* hoặc **).
- QUAN TRỌNG VỀ ANIMATION:
  * TỐI THIỂU 70% số scene phải có visual_type: "animation" (với preset "pycode" hoặc "manim").
  * {'Scene 1 của chương này là HOOK MỞ ĐẦU TOÀN BỘ VIDEO: BẮT BUỘC visual_type là "animation" do CODE AI vẽ (sóng chuyển động, mô phỏng mạng, đồ thị biến thiên...).' if is_first_ch else ''}
  * MỖI scene animation BẮT BUỘC có "visual_prompt": mô tả chi tiết hình ảnh chuyển động và màu sắc để AI chuyên code (DeepSeek-Reasoner R1 / Qwen-2.5-Coder) viết code Python Matplotlib/Manim vẽ animation tương ứng khớp với narration.
  * Mọi scene đều có "image_query" tiếng Anh (và "video_query" nếu hợp cảnh quay thực tế).

Trả về DUY NHẤT một object JSON:
{{
  "scenes": [
    {{
      "narration": "lời đọc tự nhiên có deictic narration chỉ vào hình vẽ",
      "visual_type": "animation",
      "heading": "tiêu đề ngắn trên màn hình",
      "visual_prompt": "detailed prompt for AI to write Python matplotlib/manim animation script with colors and step-by-step motion",
      "bullets": [],
      "chart": null,
      "code_language": "python",
      "algorithm": "",
      "image_query": "english keywords for background",
      "video_query": "english keywords for video footage (hoặc để trống)",
      "animation": {{"preset": "pycode"}}
    }}
  ]
}}
Chỉ trả về JSON thuần."""

        for ch_attempt in range(2):
            try:
                ch_raw = generate(ch_prompt, system=_SYSTEM, task="script")
                ch_data = _extract_json(ch_raw)
                ch_scenes_raw = ch_data.get("scenes", [])
                if not ch_scenes_raw:
                    continue
                parsed_ch_scenes = []
                for s in ch_scenes_raw:
                    if not isinstance(s, dict):
                        continue
                    if "narration" in s and isinstance(s["narration"], str):
                        s["narration"] = _clean_narration_markdown(s["narration"])
                    if "heading" in s and isinstance(s["heading"], str):
                        s["heading"] = _clean_narration_markdown(s["heading"])
                    if "bullets" in s and isinstance(s["bullets"], list):
                        s["bullets"] = [_clean_narration_markdown(b) if isinstance(b, str) else b for b in s["bullets"]]
                    try:
                        parsed_ch_scenes.append(Scene(**s))
                    except Exception as e:
                        log.warning("Bỏ qua scene lỗi: %s", e)
                if parsed_ch_scenes:
                    all_scenes.extend(parsed_ch_scenes)
                    break
            except Exception as e:
                log.warning("Thử lại viết chương %d do lỗi: %s", ch_num, e)

    if not all_scenes:
        log.warning("Viết chương phân tầng không có scene -> fallback sang prompt truyền thống")
        raw_fallback = generate(_build_prompt(topic), system=_SYSTEM, task="script")
        return _parse_script(topic, raw_fallback)

    exercises = [
        Exercise(
            question=f"Áp dụng kiến thức trong video, hãy phân tích trường hợp thực tế về {topic} khi quy mô tăng gấp 100 lần.",
            hint="Xem xét độ phức tạp tính toán và chi phí phần cứng.",
            answer="Xem phần giải thích ở phần bình luận ghim bên dưới video.",
        )
    ]

    mega_script = Script(
        topic=topic,
        title=title,
        description=description,
        tags=tags,
        scenes=all_scenes,
        exercises=exercises,
    )
    mega_script = _enforce_animation_ratio(mega_script)
    log.info(
        "Kịch bản Mega Deep-Dive '%s' hoàn tất: %d scene, %d từ lời đọc",
        mega_script.title, len(mega_script.scenes), _count_words(mega_script.scenes),
    )
    return mega_script


def _enforce_animation_ratio(script: Script) -> Script:
    """Hard Validation: Đảm bảo kịch bản đạt tối thiểu min_animation_ratio (mặc định 70%).

    Nếu LLM sinh thiếu scene animation, tự động ép các scene Pha 3 (Mổ xẻ kỹ thuật, code,
    diagram) thành visual_type: 'animation' để đảm bảo tính sinh động của video.
    """
    if not script.scenes:
        return script

    min_ratio = float(CONFIG.get("animation", {}).get("min_animation_ratio", 0.70))
    total_scenes = len(script.scenes)
    min_required = max(1, math.ceil(total_scenes * min_ratio))

    updated_scenes = list(script.scenes)
    anim_indices = {i for i, sc in enumerate(updated_scenes) if sc.visual_type == "animation"}

    # 1. Bắt buộc Scene 0 (Hook) luôn là animation
    if 0 not in anim_indices:
        sc0 = updated_scenes[0]
        v_prompt = sc0.visual_prompt or f"Motion graphic hook animation: {sc0.heading or sc0.narration[:60]}"
        updated_scenes[0] = sc0.model_copy(update={
            "visual_type": "animation",
            "visual_prompt": v_prompt,
            "animation": sc0.animation or {"preset": "motion_graphic"},
        })
        anim_indices.add(0)

    # 2. Nếu vẫn chưa đủ min_required: Ưu tiên nâng cấp Pha 3 (các scene ở giữa và có code/diagram/bullets)
    if len(anim_indices) < min_required:
        candidate_scores = []
        for i, sc in enumerate(updated_scenes):
            if i in anim_indices:
                continue
            score = 0
            if sc.visual_type in ("code", "algorithm", "diagram", "chart"):
                score += 10
            if sc.code_language:
                score += 8
            if sc.bullets:
                score += 5
            pos_ratio = i / total_scenes
            if 0.20 <= pos_ratio <= 0.85:
                score += 6
            candidate_scores.append((score, i))

        candidate_scores.sort(key=lambda x: x[0], reverse=True)

        for _, idx in candidate_scores:
            if len(anim_indices) >= min_required:
                break
            sc = updated_scenes[idx]
            v_prompt = sc.visual_prompt or f"Dynamic tech motion visualization for: {sc.heading or sc.narration[:80]}"
            updated_scenes[idx] = sc.model_copy(update={
                "visual_type": "animation",
                "visual_prompt": v_prompt,
                "animation": sc.animation or {"preset": "motion_graphic"},
            })
            anim_indices.add(idx)

    actual_ratio = len(anim_indices) / total_scenes
    log.info(
        "Hard Validation Animation: %d/%d scene (%.1f%%, yêu cầu >=%.0f%%)",
        len(anim_indices), total_scenes, actual_ratio * 100, min_ratio * 100,
    )
    return script.model_copy(update={"scenes": updated_scenes})


def write_script(topic: str, series_context: dict | None = None) -> Script:
    duration = int(CONFIG.get("target_duration_seconds", 300))
    mode = CONFIG.get("active_mode", "long")

    # Lấy tư liệu thực tế & test case qua TinyFish nếu khả dụng
    research_context = ""
    try:
        from .tinyfish_client import is_available, research_topic_context

        if is_available():
            log.info("TinyFish: Đang trinh sát dữ liệu thực tế cho chủ đề '%s'...", topic)
            research_context = research_topic_context(topic)
            if research_context:
                log.info("TinyFish: Đã thu thập tư liệu thực tế cho kịch bản.")
    except Exception as e:  # noqa: BLE001
        log.debug("TinyFish topic research loi: %s", e)

    # Nếu mode là mega hoặc thời lượng >= 600s (~10 phút trở lên) -> dùng kịch bản phân tầng đa chương
    if mode == "mega" or duration >= 600:
        return _write_mega_script(topic, duration, research_context=research_context)

    approx_words = int(duration / 60 * 130)
    # Ngưỡng tối thiểu: long cần ~75% mục tiêu; short cần ~80% mục tiêu để kịch bản bám sát thời lượng (FIX-08).
    min_words = int(approx_words * 0.75) if mode != "short" else int(approx_words * 0.80)
    min_scenes = max((duration // 60) * 2, 8) if mode != "short" else max(int(duration / 25), 3)

    script: Script | None = None
    for attempt in range(2):  # thử tối đa 2 lần nếu kịch bản quá ngắn/thiếu scene
        raw = generate(
            _build_prompt(topic, research_context=research_context, series_context=series_context),
            system=_SYSTEM,
            task="script",
        )
        script = _parse_script(topic, raw)
        words = _count_words(script.scenes)
        if words >= min_words and len(script.scenes) >= min_scenes:
            break
        log.warning(
            "Kịch bản lần %d chưa đạt (%d từ / %d scene, cần >=%d từ, >=%d scene), thử lại.",
            attempt + 1, words, len(script.scenes), min_words, min_scenes,
        )

    assert script is not None

    # Lượt riêng tối ưu tiêu đề: sinh 5 phương án, tự chọn tốt nhất.
    # Không ảnh hưởng kịch bản; thất bại thì giữ title gốc.
    optimized = _optimize_title(script)
    if optimized and optimized != script.title:
        script = script.model_copy(update={"title": optimized})

    # Nếu thuộc Series, tự động gắn tiền tố [Tập X/Y] vào tiêu đề
    if series_context and script:
        ep_num = series_context.get("episode_num")
        total_eps = series_context.get("total_episodes")
        prefix = f"[Tập {ep_num}/{total_eps}]"
        if prefix not in script.title:
            script = script.model_copy(update={"title": f"{prefix} {script.title}"})

    # Hard Validation: Bắt buộc kịch bản đạt tỉ lệ animation tối thiểu >= 70%
    script = _enforce_animation_ratio(script)

    log.info(
        "Kịch bản '%s' có %d scene, %d bài toán thực tế, %d từ lời đọc",
        script.title, len(script.scenes), len(script.exercises), _count_words(script.scenes),
    )
    return script
