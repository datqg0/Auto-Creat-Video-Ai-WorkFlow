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
    "Bạn là biên kịch cho kênh YouTube giải thích công nghệ bằng hình ảnh trực quan "
    "(giống 3Blue1Brown, Kurzgesagt). Bạn luôn trả về JSON hợp lệ, không kèm giải thích."
)


def _as_question(title: str) -> str:
    """Đảm bảo tiêu đề ở dạng câu hỏi (kết thúc bằng '?')."""
    t = (title or "").strip().rstrip(".!…")
    if not t:
        return t
    if t.endswith("?"):
        return t
    return t + "?"


def _title_case(title: str) -> str:
    """Viết hoa chữ cái đầu mỗi từ, giữ nguyên phần còn lại (giữ acronym AI/GPU)."""
    def cap(w: str) -> str:
        return w[:1].upper() + w[1:] if w else w
    return " ".join(cap(w) for w in (title or "").split(" "))


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
      "camera"(zoom/pan toàn cảnh: {{"anim":"camera","zoom":1.8,"cx":"0.5W","cy":"0.4H","run_time":1.2}} — KHÔNG cần target),
      "transform"/"morph"(biến hình A->B: {{"anim":"transform","target":"c","to":"b","run_time":1.0}} — cần "to" là id đích),
      "move_along"(chạy 1 dot dọc theo graph/parametric: {{"anim":"move_along","target":"d","path":"p","trace":true,"run_time":2.0}} — "path" là id graph/parametric, "trace":true vẽ dần nét ngay dưới điểm chạy),
      "vmorph"(biến hình THỬeC theo đỉnh: {{"anim":"vmorph","target":"pg","from":"c","to":"b","run_time":1.5}} — "target" phải là polygon, "from"/"to" là id circle/rect/polygon; mượt hơn "transform").
    - Hãy sáng tạo: kết hợp nhiều phần tử + bước để "kể" ý tưởng bằng chuyển động,
      ví dụ vẽ trục -> kéo đồ thị (glow) -> cho dot chạy dọc đường cong -> zoom camera vào -> nhấn mạnh công thức LaTeX.
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

QUAN TRỌNG về hình ảnh (video phải THẬT NHIỀU hình ảnh & animation, không được nhàm):
- Đa dạng visual_type: dùng ÍT NHẤT 5 loại khác nhau, KHÔNG để 2 scene bullets liên tiếp.
- BẮT BUỘC có TỐI THIỂU 3-5 scene "animation" rải đều trong video (hàm số, mạng neural,
  số liệu, thuật toán, quy trình) để video sinh động như 3Blue1Brown.
- BẮT BUỘC có ÍT NHẤT 2 scene animation preset "pycode" (tự viết code Python/matplotlib
  vẽ hình ảnh minh họa) rải ở các phần khác nhau của video — để mỗi video có tối thiểu 2
  hình minh họa do code sinh ra, không chỉ 1.
- NÊN có ÍT NHẤT 1-2 animation "custom" tự thiết kế (không chỉ dùng preset có sẵn) để
  minh họa đúng ý tưởng cốt lõi của video một cách độc đáo, sinh động.
- Với scene animation "custom" hoặc "pycode": nếu đã TỰ ĐẶT chữ tiêu đề bên trong (một
  object text/formula ở phía trên, hoặc dòng title trong code), thì để "heading" TRỐNG ("")
  để tránh CHỒNG CHỮ (2 lớp tiêu đề đè lên nhau).
- MỌI scene (trừ code) đều PHẢI có "image_query": 2-5 từ khóa TIẾNG ANH mô tả ảnh minh họa nền
  cụ thể, sinh động (ví dụ "neural network brain glowing", "data center servers blue",
  "encryption padlock circuit", "quantum computer chip"). Không để trống.
- Ưu tiên hình ảnh trực quan hơn chữ: mỗi ý nên gắn với 1 hình ảnh hoặc animation minh họa.
- "video_query": với các scene hợp với CẢNH QUAY THỰC (data center, con chip, người dùng
  điện thoại/laptop, robot, thành phố, mạch điện, phòng lab...), thêm 2-4 từ khóa TIẾNG ANH
  để tải footage VIDEO thật làm nền động (ví dụ "data center servers", "person using smartphone",
  "circuit board macro", "city traffic night"). Nếu scene trừu tượng/toán học thì để trống "".
  Nên có 3-6 scene có video_query rải đều để video trực quan, sinh động hơn ảnh tĩnh.

Trả về DUY NHẤT một object JSON theo schema:
{{
  "title": "tiêu đề là MỘT CÂU HỎI mà video sẽ trả lời (kết thúc bằng dấu ?), gây tò mò, dưới 70 ký tự",
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
      "video_query": "từ khóa footage video tiếng Anh (hoặc để trống)",
      "animation": null
    }}
  ],
  "exercises": [
    {{
      "question": "một bài toán/tình huống THỰC TẾ để người xem tự giải",
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
- "title" BẮT BUỘC là MỘT CÂU HỎI (kết thúc bằng "?") mà nội dung video sẽ giải đáp;
  ưu tiên dạng "Tại sao...?", "Làm thế nào...?", "Điều gì xảy ra khi...?", "Có thật là...?".
  Scene HOOK mở đầu phải đặt lại đúng câu hỏi này, và scene TỔNG KẾT phải trả lời rõ nó.
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
- Bố cục DỌC 9:16: chữ TO, RẤT ÍT chữ mỗi màn hình để không tràn khung trên điện thoại.

CẤU TRÚC SHORT (rút gọn từ flow giáo dục, giữ nhịp nhanh):
1. HOOK cực mạnh trong 2 giây đầu — một câu hỏi sốc hoặc con số gây tò mò.
2. VẤN ĐỀ — nêu nhanh điều bất ngờ/khó hiểu.
3. GIẢI THÍCH TRỰC QUAN — 1-2 scene cốt lõi, dùng animation hoặc ảnh mạnh để "aha".
4. ĐIỂM CHỐT — insight/con số đáng nhớ nhất.
5. CALL-TO-ACTION — "Theo dõi để xem phần tiếp theo" hoặc câu hỏi mở kéo comment.

Yêu cầu hình ảnh cho Short (khung DỌC hẹp, tránh tràn chữ):
- Mỗi scene PHẢI có "image_query" 2-5 từ khóa TIẾNG ANH, ảnh nổi bật, tương phản cao.
- Nên có 1-2 scene "animation" (counter con số, function, hoặc steps) để bắt mắt.
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
    if start == -1 or end == -1:
        raise ValueError("Không tìm thấy JSON trong output LLM")
    return json.loads(text[start : end + 1])


def write_script(topic: str) -> Script:
    raw = generate(_build_prompt(topic), system=_SYSTEM)
    data = _extract_json(raw)

    scenes: list[Scene] = []
    for s in data.get("scenes", []):
        if not isinstance(s, dict):
            continue
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

    script = Script(
        topic=topic,
        title=_title_case(_as_question(data.get("title", topic)))[:100],
        description=data.get("description", ""),
        tags=data.get("tags", []),
        scenes=scenes,
        exercises=exercises,
    )
    log.info(
        "Kịch bản '%s' có %d scene, %d bài toán thực tế",
        script.title, len(scenes), len(exercises),
    )
    return script
