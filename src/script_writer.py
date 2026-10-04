"""LLM viết kịch bản video chia theo scene, mỗi scene kèm visual spec."""
from __future__ import annotations

import json
import logging
import re

from .config import CONFIG
from .llm import generate
from .models import Exercise, Scene, Script

log = logging.getLogger(__name__)

_SYSTEM = (
    "Bạn là biên kịch trưởng kiêm chuyên gia sư phạm công nghệ cho kênh YouTube giáo dục đỉnh cao "
    "(theo phong cách 3Blue1Brown, Veritasium, Kurzgesagt). Bạn áp dụng triệt để Nguyên lý Feynman: "
    "biến những khái niệm khoa học máy tính, toán học và kiến trúc hệ thống phức tạp nhất trở nên "
    "cực kỳ trực quan, dễ hiểu, sáng tỏ qua ẩn dụ đời thường và mô phỏng thị giác chuẩn xác. "
    "Mọi ví dụ bạn đưa ra đều thực chiến, có số liệu cụ thể. Bạn luôn trả về JSON hợp lệ, không kèm giải thích ngoài."
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


def _build_prompt(topic: str) -> str:
    lang = CONFIG.get("language", "vi")
    lang_name = "tiếng Việt" if lang == "vi" else "English"
    duration = int(CONFIG.get("target_duration_seconds", 300))
    mode = CONFIG.get("active_mode", "long")
    # Tốc độ đọc giáo dục thư thái: ~130 từ/phút tiếng Việt (kèm khoảng nghỉ giữa các câu để người xem kịp ngấm).
    approx_words = int(duration / 60 * 130)
    min_words = int(approx_words * 0.85)

    if mode == "short":
        return _build_short_prompt(topic, lang_name, duration, min_words, approx_words)

    minutes = max(duration // 60, 1)
    n_scenes_min = max(minutes * 2, 8)   # ~2 scene/phút -> nhịp điệu vừa vặn, không lê thê
    n_scenes_max = n_scenes_min + 4

    return f"""Viết kịch bản cho video YouTube dài ĐỦ {duration} giây (~{minutes} phút) về chủ đề:
"{topic}"

Ngôn ngữ: {lang_name}. YÊU CẦU ĐỘ DÀI & NHỊP ĐIỆU (BẮT BUỘC):
- Tổng lời đọc (cộng dồn tất cả narration) tối thiểu {min_words} từ, mục tiêu ~{approx_words} từ (nhịp đọc thư thái ~130 từ/phút).
- Chia thành {n_scenes_min}-{n_scenes_max} scene, phủ ĐỦ 10 bước cấu trúc bên dưới.
- Mỗi scene narration 3-6 câu (~50-90 từ), câu ngắn gọn (8-16 từ), TUYỆT ĐỐI không viết câu dài lê thê hay đọc dồn dập.

NGUYÊN TẮC SƯ PHẠM TRỰC QUAN (FEYNMAN & 3BLUE1BROWN — CỰC KỲ QUAN TRỌNG):
1. ẨN DỤ ĐỜI THƯỜNG TRƯỚC, THUẬT NGỮ SAU:
   - TUYỆT ĐỐI KHÔNG mở đầu bài học bằng định nghĩa khô khan hay công thức toán học.
   - BẮT BUỘC mở đầu bằng một Ẩn dụ thực tế gần gũi (ví dụ: giao thông kẹt xe, xếp hàng thanh toán siêu thị, thủ thư xếp sách vào ngăn kéo tủ, gửi bưu phẩm...).
   - Cho người xem "cảm nhận bằng trực giác" bản chất của vấn đề trước khi gán nhãn thuật ngữ chuyên ngành.
2. VÍ DỤ THỰC CHIẾN CỰC CHUẨN (ROCK-SOLID EXAMPLES):
   - Mọi ví dụ giải thích BẮT BUỘC gắn với ứng dụng thực tế từ các hệ thống lớn: Google, Netflix, Shopee Flash Sale, Git, Ngân hàng, Hệ điều hành...
   - Nêu rõ: Đầu vào cụ thể (Input) -> Quá trình xử lý từng bước -> Đầu ra (Output).
   - Nêu bật con số định lượng: "Nếu làm cách ngây thơ mất 10 giây; với thuật toán này chỉ mất 2 mili-giây".
3. TƯ DUY NÚT THẮT & AHA MOMENT:
   - Đặt câu hỏi: "Tại sao cách làm bình thường lại bế tắc?" -> Nêu bật ý tưởng thông minh tháo gỡ bế tắc.

HOOK 5 GIÂY ĐẦU (QUYẾT ĐỊNH GIỮ CHÂN NGƯỜI XEM):
- Scene 1 BẮT BUỘC có visual_type: "animation" (preset "pycode" hoặc "manim").
- Mở đầu video ngay từ giây đầu tiên bằng một animation ĐỘNG do CODE PYTHON trực quan vẽ (sóng xung kích, mô phỏng mạng, đồ thị biến thiên, nghịch lý trực quan...). TUYỆT ĐỐI không mở đầu bằng slide tĩnh hay chữ đơn điệu.
- Câu ĐẦU TIÊN của narration scene 1 phải là một cú móc mạnh: một con số gây sốc, một nghịch lý, hoặc một câu hỏi phản trực giác khiến người xem không thể rời mắt.
- TUYỆT ĐỐI không mở đầu bằng "Xin chào", "Trong video này", "Hôm nay chúng ta". Vào thẳng vấn đề!

CẤU TRÚC 10 BƯỚC BẮT BUỘC:
1. HOOK — câu hỏi/tình huống gây tò mò trong 10 giây đầu, hứa hẹn giá trị.
2. ĐẶT BÀI TOÁN — nêu rõ vấn đề cần giải quyết, vì sao nó khó và quan trọng.
3. ẨN DỤ TRỰC GIÁC — hình tượng hóa bằng ẩn dụ đời thường để người xem "cảm" được ngay.
4. Ý TƯỞNG ĐỘT PHÁ — ý tưởng cốt lõi giải quyết bài toán ("Aha moment").
5. VÍ DỤ TỪNG BƯỚC — đi qua một ví dụ cụ thể, từng bước một với số liệu thật (nên dùng diagram/animation steps).
6. CƠ CHẾ KỸ THUẬT / CODE — hình thức hóa bằng cấu trúc dữ liệu, thuật toán hoặc code cụ thể.
7. DEMO ỨNG DỤNG THỰC TẾ — cho thấy các hệ thống lớn áp dụng ra sao (kèm con số hiệu năng).
8. BÀI TẬP / THỬ THÁCH — đặt 1 câu hỏi/tình huống thực tế kích thích tư duy người xem.
9. TỔNG KẾT — chốt lại bản chất trong 1 câu đắt giá + kêu gọi đăng ký kênh.
10. MỞ SANG CHỦ ĐỀ KẾ TIẾP — gợi mở câu hỏi tiếp theo để giữ chân người xem xem video sau.

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
    - Hãy sáng tạo: kết hợp nhiều phần tử + bước để "kể" ý tưởng bằng chuyển động,
      ví dụ vẽ trục -> kéo đồ thị (glow) -> cho dot chạy dọc đường cong -> nhấn mạnh công thức LaTeX.
  * TỰ VIẾT CODE Python (matplotlib) để vẽ animation phức tạp mà preset/custom chưa làm được:
    {{"preset": "pycode", "code": "..."}}
    - "code" là code Python DÙNG matplotlib (đã import sẵn backend Agg). Có sẵn các biến:
      WIDTH, HEIGHT, FPS, DURATION (giây), OUT_PATH (đường dẫn mp4 phải lưu vào).
    - BẮT BUỘC lưu animation vào file OUT_PATH dạng mp4, ví dụ:
      dùng matplotlib.animation.FuncAnimation rồi ani.save(OUT_PATH, fps=FPS, writer="ffmpeg").
    - Đặt figure đúng khung: fig = plt.figure(figsize=(WIDTH/100, HEIGHT/100), dpi=100).
    - Số frame nên = int(DURATION * FPS) để khớp thời lượng lời đọc.
    - CHỈ dùng matplotlib + numpy để VẼ. Code chạy trong sandbox KHÔNG có mạng, KHÔNG có
      biến môi trường/secret; đừng đọc file ngoài, đừng gọi mạng — sẽ vô ích.
    - Nếu code lỗi/quá lâu, hệ thống tự chuyển về ảnh tĩnh -> hãy viết code gọn, chắc chắn chạy.

    {{"preset": "manim", "code": "..."}}
    - "code" là code Python DÙNG thư viện manim để tạo animation chất lượng cao (kiểu 3Blue1Brown).
    - BẮT BUỘC định nghĩa MỘT class kế thừa Scene với method construct(self), ví dụ:
      "class AIScene(Scene):\\n    def construct(self):\\n        t = Text('Xin chào'); self.play(Write(t)); self.wait(1)"
    - Đã import sẵn `from manim import *`. Có sẵn biến DURATION (giây) để canh nhịp;
      độ phân giải/fps do hệ thống cấu hình — KHÔNG tự set config.
    - LaTeX chỉ cài BẢN SLIM (texlive-base + recommended) -> ƯU TIÊN Text/MarkupText;
      TRÁNH MathTex/Tex phức tạp (dễ lỗi biên dịch). Công thức đơn giản mới dùng MathTex.
    - Giữ animation NGẮN GỌN (vài giây, ít object) để không bị timeout khi render.
    - Code chạy trong sandbox KHÔNG có mạng, KHÔNG có secret; đừng đọc file ngoài/gọi mạng.
    - Nếu manim chưa cài hoặc render lỗi/timeout, hệ thống tự fallback sang pycode/ảnh tĩnh.
    - CHỈ dùng preset "manim" cho 1-2 scene quan trọng nhất (render manim CHẬM & nặng).

QUAN TRỌNG VỀ NHỊP VĂN & GIỌNG ĐỌC (THƯ THÁI, DỄ HIỂU, TỰ NHIÊN):
- NHỊP ĐỌC ĐÀM THOẠI & CÂU NGẮN (8-16 TỪ): Viết câu ngắn gọn, gãy gọn, giàu tính đàm thoại. Tránh câu phức dài ngoằng.
  Ngắt nhịp tự nhiên để bộ đọc TTS đọc thư thái, có điểm nhấn, giúp người nghe kịp ngấm kiến thức.
- LỜI ĐỌC TỰ NHIÊN, VĂN BẢN THUẦN: Dùng ngôn ngữ bình dân, gợi hình, so sánh trực quan.
  TUYỆT ĐỐI KHÔNG dùng dấu sao (* hoặc **) hoặc gạch dưới để bôi đậm từ, vì bộ đọc giọng nói TTS sẽ phát âm thành chữ "sao" hoặc "hoa thị".
- NGUYÊN TẮC DỄ HIỂU: Nói như đang trò chuyện với một người bạn thông minh nhưng mới bắt đầu tìm hiểu chủ đề này.

QUAN TRỌNG VỀ HÌNH ẢNH & VISUAL PROMPT (BẮT BUỘC ĐẠT ĐỘ CHUẨN XÁC TUYỆT ĐỐI):
- BẮT BUỘC có ÍT NHẤT 60% - 80% số scene có visual_type: "animation" do code AI viết (preset "pycode" hoặc "manim").
- Scene 1 (HOOK) BẮT BUỘC có visual_type: "animation" do code AI vẽ ngay từ giây đầu tiên.
- KHÔNG để 2 scene bullets hoặc 2 ảnh tĩnh liên tiếp. Phải xen kẽ: Animation code -> B-roll video -> Animation code -> Slide trực quan.
- TIÊU CHUẨN VÀNG CHO "visual_prompt" (CỰC KỲ CHI TIẾT ĐỂ AI GEN CODE CHUẨN XÁC):
  * TUYỆT ĐỐI KHÔNG viết visual_prompt chung chung như "vẽ thuật toán" hay "mô phỏng hệ thống".
  * BẮT BUỘC mô tả 3 yếu tố:
    1. DỮ LIỆU CỤ THỂ: Mảng số cụ thể [5, 2, 8, 1, 9], cây nhị phân 3 tầng, 4 node A-B-C-D hay cấu trúc bảng hash...
    2. CHUYỂN ĐỘNG TỪNG BƯỚC: Con trỏ di chuyển qua đâu, phần tử nào đổi màu (Cyan = chờ, Vàng/Hồng = đang xét, Xanh neon = đã khớp/thành công), mũi tên gửi gói tin `send(a, b)` như thế nào.
    3. NHỊP ĐIỆU DIỄN HOẠT: Di chuyển mượt mà, có khoảng dừng để người xem nhìn rõ cơ chế hoạt động.
- ƯU TIÊN preset "pycode" (Matplotlib) hoặc "manim" để mọi khái niệm kỹ thuật đều được mô phỏng sinh động bằng code thực thi.
- NÊN có ÍT NHẤT 1-2 animation "custom" tự thiết kế (không chỉ dùng preset có sẵn) để minh họa đúng ý tưởng cốt lõi của video.
- Với scene animation "custom" hoặc "pycode": nếu đã TỰ ĐẶT chữ tiêu đề bên trong, thì để "heading" TRỐNG ("") để tránh CHỒNG CHỮ.
- MỌI scene (trừ code) đều PHẢI có "image_query": 2-5 từ khóa TIẾNG ANH mô tả ảnh minh họa nền cụ thể, sinh động (ví dụ "neural network brain glowing", "data center servers blue", "encryption padlock circuit", "quantum computer chip"). Không để trống.
- "video_query": với các scene hợp với CẢNH QUAY THỰC (data center, con chip, người dùng điện thoại/laptop, robot, thành phố, mạch điện, phòng lab...), thêm 2-4 từ khóa TIẾNG ANH để tải footage VIDEO thật làm nền động. Nên có 3-6 scene có video_query rải đều để video trực quan.

Trả về DUY NHẤT một object JSON theo schema:
{{
  "title": "tiêu đề gây tò mò mạnh, dưới 70 ký tự (xem quy tắc tiêu đề bên dưới)",
  "description": "mô tả 3-4 câu cho YouTube, có hashtag ở cuối",
  "tags": ["tag1", "tag2", "..."],
  "scenes": [
    {{
      "narration": "lời đọc tự nhiên, câu ngắn dưới 18 từ (văn bản thuần, TUYỆT ĐỐI không dùng dấu * hoặc **)",
      "visual_type": "bullets",
      "heading": "tiêu đề ngắn hiển thị trên màn hình",
      "visual_prompt": "prompt cực kỳ chi tiết về dữ liệu và chuyển động từng bước để AI viết code Python/Manim vẽ chuẩn xác",
      "bullets": ["ý 1", "ý 2"],
      "chart": null,
      "code_language": "python",
      "algorithm": "",
      "image_query": "từ khóa ảnh tiếng Anh",
      "video_query": "từ khóa footage video tiếng Anh (hoặc để trống)",
      "animation": null
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

BÀI TẬP VÍ DỤ (BẮT BUỘC): tạo mảng "exercises" gồm ĐÚNG 1 bài toán/tình huống THỰC TẾ
tiêu biểu nhất, để người xem tự luyện ở CUỐI video. Bài phải cụ thể, gắn với ứng dụng
đời thực (con số, tình huống công việc/cuộc sống), kèm "hint" ngắn và "answer" gợi hướng làm.
KHÔNG hỏi lý thuyết suông. CHỈ 1 bài — không tạo nhiều bài tập.

Lưu ý:
- "title" phải CHỌN MỘT trong các dạng sau (mỗi dạng đều tăng CTR theo nghiên cứu YouTube):
  A. Câu hỏi gây sốc/tò mò: "Tại sao mọi website đều đang bị tấn công ngay lúc này?"
  B. Con số + lợi ích rõ ràng: "5 phút hiểu thuật toán mà mọi Big Tech đều dùng"
  C. Khoảng cách tò mò (không tiết lộ đáp án): "Thứ ẩn trong mọi video YouTube bạn xem"
  D. Cổ phần + khẩn cấp: "Lỗi này đã làm mất 1 tỷ USD dữ liệu — và bạn đang mắc nó"
  E. Phản trực giác: "GPU thực ra chậm hơn CPU — nhưng đây là lý do nó thắng"
  TUYỆT ĐỐI KHÔNG dùng: "Giới thiệu về...", "Tìm hiểu...", "Hướng dẫn...".
  Scene HOOK mở đầu phải đặt lại đúng câu hỏi/tuyên bố này, và scene TỔNG KẾT phải trả lời rõ nó.
- narration phải liền mạch, kể chuyện, KHÔNG đọc gạch đầu dòng, KHÔNG quá ngắn.
- Scene đầu là HOOK (visual_type "animation" do code AI vẽ), scene gần cuối là TỔNG KẾT ("quote"),
  scene cuối cùng là MỞ SANG VIDEO TIẾP THEO (gợi mở + call-to-action đăng ký).
- Bám sát 10 bước cấu trúc theo đúng thứ tự; heading mỗi scene nên phản ánh bước đang ở.
- Với "chart", số liệu hợp lý, labels/values cùng độ dài.
- Chỉ trả JSON, không markdown, không ```."""


def _build_short_prompt(
    topic: str, lang_name: str, duration: int, min_words: int, approx_words: int
) -> str:
    """Prompt cho YouTube Short: dọc 9:16, tối đa 3 phút (180s), nhịp chậm rãi, dễ hiểu, hook cực mạnh."""
    n_scenes_min = max(int(duration / 25), 4)
    n_scenes_max = max(int(duration / 15), 6)
    return f"""Viết kịch bản cho một YouTube SHORT (video DỌC 9:16, thời lượng mục tiêu ~{duration} giây, TUYỆT ĐỐI không vượt quá 180 giây / 3 phút) về chủ đề:
"{topic}"

Ngôn ngữ: {lang_name}. YÊU CẦU BẮT BUỘC CHO SHORT (DỄ HIỂU & CHUẨN XÁC):
- Tổng lời đọc khoảng {min_words}-{approx_words} từ (nhịp đọc thư thái, rõ ràng, TUYỆT ĐỐI không vượt quá 180s).
- Chia thành {n_scenes_min}-{n_scenes_max} scene ngắn gọn, mỗi scene narration 1-3 câu ngắn (10-20 từ), nhịp điệu tự nhiên, dễ ngấm.
- Bố cục DỌC 9:16: chữ TO, RẤT ÍT chữ mỗi màn hình để không tràn khung trên điện thoại.
- DỄ HIỂU & TRỰC QUAN: Mở đầu bằng một câu hỏi sốc hoặc ví dụ đời thường, giải thích bản chất bằng trực giác trước khi nêu giải pháp kỹ thuật.

CẤU TRÚC SHORT:
1. HOOK cực mạnh trong 2 giây đầu — một câu hỏi sốc hoặc con số gây tò mò.
2. ẨN DỤ / VẤN ĐỀ — nêu nhanh tình huống bất ngờ hoặc ví dụ đời thực dễ hiểu.
3. GIẢI THÍCH TRỰC QUAN — các scene cốt lõi giải thích cơ chế, dùng animation code trực quan để tạo cảm giác "aha".
4. ĐIỂM CHỐT — insight hoặc con số đáng nhớ nhất.
5. CALL-TO-ACTION — câu hỏi mở kéo comment hoặc kêu gọi theo dõi phần tiếp theo.

Yêu cầu hình ảnh cho Short (khung DỌC hẹp, tránh tràn chữ):
- MỖI scene animation PHẢI có "visual_prompt" chi tiết để AI gen code Python/Manim vẽ chuyển động chính xác.
- Mỗi scene PHẢI có "image_query" 2-5 từ khóa TIẾNG ANH, ảnh nổi bật, tương phản cao.
- Nên có 2-3 scene "animation" (counter con số, function, hoặc steps) để bắt mắt.
- Ưu tiên visual_type: "title", "quote", "animation"; hạn chế "bullets".
- "heading" TỐI ĐA 4-5 từ (chữ to, dễ tràn nếu dài). Mỗi bullet TỐI ĐA 6-8 từ, tối đa 3 bullet/scene.
- KHÔNG viết câu dài trong heading/bullets; để câu dài cho narration.
- KHÔNG dùng visual_type "code", "chart", "diagram" (khó đọc trên khung dọc).

Trả về DUY NHẤT một object JSON theo schema:
{{
  "title": "tiêu đề là MỘT CÂU HỎI giật tít mà Short sẽ trả lời (kết thúc bằng ?), dưới 60 ký tự",
  "description": "1-2 câu + hashtag (#Shorts và 3-4 hashtag chủ đề) ở cuối",
  "tags": ["shorts", "tag2", "..."],
  "scenes": [
    {{
      "narration": "lời đọc ngắn, dứt khoát",
      "visual_type": "title",
      "heading": "chữ to hiển thị",
      "visual_prompt": "prompt mô tả animation hoặc hình ảnh visual trực quan",
      "bullets": [],
      "chart": null,
      "code_language": "python",
      "algorithm": "",
      "image_query": "từ khóa ảnh tiếng Anh",
      "video_query": "từ khóa footage video tiếng Anh (hoặc để trống)",
      "animation": null
    }}
  ],
  "exercises": [
    {{
      "question": "một bài toán/tình huống THỰC TẾ ngắn để người xem tự giải",
      "hint": "",
      "answer": ""
    }}
  ]
}}

BÀI TẬP VÍ DỤ (BẮT BUỘC): tạo mảng "exercises" gồm ĐÚNG 1 bài toán/tình huống THỰC TẾ
ngắn gọn tiêu biểu, đặt ở cuối. Bài cụ thể, gắn ứng dụng đời thực. CHỈ 1 bài.

Lưu ý:
- "title" BẮT BUỘC là MỘT CÂU HỎI (kết thúc bằng "?") mà Short sẽ giải đáp.
  Scene HOOK phải đặt lại đúng câu hỏi này, và điểm chốt phải trả lời rõ nó.
- Scene đầu = HOOK, scene cuối = CALL-TO-ACTION.
- Tổng lời đọc phải NGẮN để lọt dưới {duration} giây. Ưu tiên súc tích hơn đầy đủ.
- Chỉ trả JSON, không markdown, không ```."""


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
    t = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
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


def _write_mega_script(topic: str, duration: int) -> Script:
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

    outline_prompt = f"""Bạn là đạo diễn và biên kịch trưởng cho kênh YouTube giải thích công nghệ chuyên sâu (kiểu 3Blue1Brown, Fireship, ByteByteGo).
Chúng ta đang sản xuất video MASTERCLASS SIÊU DÀI ĐẶC BIỆT (~{minutes} phút, mục tiêu ~{approx_words} từ lời đọc) về chủ đề:
"{topic}"

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
- Tạo đúng {n_scenes} scene liền mạch, giải thích sâu sắc, lời đọc tự nhiên (~{total_words_target_per_chapter} từ cho cả chương).
- Mỗi scene narration dài 3-6 câu (~60-100 từ), câu ngắn dưới 20 từ, TUYỆT ĐỐI không dùng dấu hoa thị/sao (* hoặc **).
- QUAN TRỌNG VỀ ANIMATION:
  * TỐI THIỂU 70% số scene phải có visual_type: "animation" (với preset "pycode" hoặc "manim").
  * {'Scene 1 của chương này là HOOK MỞ ĐẦU TOÀN BỘ VIDEO: BẮT BUỘC visual_type là "animation" do CODE AI vẽ (sóng chuyển động, mô phỏng mạng, đồ thị biến thiên...).' if is_first_ch else ''}
  * MỖI scene animation BẮT BUỘC có "visual_prompt": mô tả chi tiết hình ảnh chuyển động để AI chuyên code (DeepSeek-Reasoner R1 / Qwen-2.5-Coder) viết code Python Matplotlib/Manim vẽ animation tương ứng.
  * Mọi scene đều có "image_query" tiếng Anh (và "video_query" nếu hợp cảnh quay thực tế).

Trả về DUY NHẤT một object JSON:
{{
  "scenes": [
    {{
      "narration": "lời đọc tự nhiên, câu ngắn gãy gọn",
      "visual_type": "animation",
      "heading": "tiêu đề ngắn trên màn hình",
      "visual_prompt": "detailed prompt for AI to write Python matplotlib/manim animation script",
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
    log.info(
        "Kịch bản Mega Deep-Dive '%s' hoàn tất: %d scene, %d từ lời đọc",
        mega_script.title, len(mega_script.scenes), _count_words(mega_script.scenes),
    )
    return mega_script


def write_script(topic: str) -> Script:
    duration = int(CONFIG.get("target_duration_seconds", 300))
    mode = CONFIG.get("active_mode", "long")

    # Nếu mode là mega hoặc thời lượng >= 600s (~10 phút trở lên) -> dùng kịch bản phân tầng đa chương
    if mode == "mega" or duration >= 600:
        return _write_mega_script(topic, duration)

    approx_words = int(duration / 60 * 130)
    # Ngưỡng tối thiểu: long cần ~75% mục tiêu; short cần ~65% mục tiêu để kịch bản bám sát thời lượng.
    min_words = int(approx_words * 0.75) if mode != "short" else int(approx_words * 0.65)
    min_scenes = max((duration // 60) * 2, 8) if mode != "short" else max(int(duration / 25), 3)

    script: Script | None = None
    for attempt in range(2):  # thử tối đa 2 lần nếu kịch bản quá ngắn/thiếu scene
        raw = generate(_build_prompt(topic), system=_SYSTEM, task="script")
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

    log.info(
        "Kịch bản '%s' có %d scene, %d bài toán thực tế, %d từ lời đọc",
        script.title, len(script.scenes), len(script.exercises), _count_words(script.scenes),
    )
    return script
