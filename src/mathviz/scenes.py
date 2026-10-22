"""Scene dựng sẵn (presets) cho các concept tech/math hay gặp.

Mỗi hàm nhận nội dung + duration và trả về 1 ``Scene`` đã dựng timeline sẵn.
Pipeline chỉ cần chọn preset theo kiểu scene rồi gọi ``fit_duration`` + render.

Đây là lớp "cơ động cao": muốn thêm kiểu mới chỉ cần viết thêm 1 hàm ở đây,
tái dùng toàn bộ primitives + animation của thư viện.
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

from .core import Scene
from .objects import (
    Axes,
    BarChart,
    Circle,
    FunctionGraph,
    Group,
    Line,
    NeuralNet,
    Rect,
    Stickman,
    Text,
)
from .anims import (
    CountUp,
    DrawLine,
    FadeIn,
    GrowFromCenter,
    Signal,
    Write,
)
from .theme import THEME


def _title_block(scene: Scene, title: str, subtitle: str = "") -> None:
    """Thêm tiêu đề trên cùng cho scene (dùng chung nhiều preset)."""
    W = THEME.width
    t = Text(title, (W / 2, 120), size=64, color=THEME.text, anchor="mm")
    scene.play(FadeIn(t, shift=30, run_time=0.7))
    if subtitle:
        s = Text(subtitle, (W / 2, 200), size=34, color=THEME.muted, anchor="mm", bold=False)
        scene.play(FadeIn(s, run_time=0.5))


# --------------------------------------------------------------- function plot
def function_scene(
    title: str,
    fn: Callable[[float], float],
    x_range: tuple[float, float] = (-6.28, 6.28),
    y_range: tuple[float, float] = (-1.4, 1.4),
    subtitle: str = "",
    color: Optional[str] = None,
    duration: Optional[float] = None,
) -> Scene:
    """Vẽ đồ thị 1 hàm số, curve được 'kéo' ra từ trái sang phải."""
    scene = Scene(duration=duration)
    W, H = THEME.width, THEME.height
    _title_block(scene, title, subtitle)
    axes = Axes(
        box=(W * 0.14, 300, W * 0.86, H - 140),
        x_range=x_range,
        y_range=y_range,
        grid=True,
    )
    scene.play(FadeIn(axes, run_time=0.6))
    graph = FunctionGraph(axes, fn, color=color or THEME.accent, width=6)
    scene.play(DrawLine(graph, run_time=2.2))
    # điểm chạy dọc theo curve để nhấn mạnh
    dot = Circle((0, 0), 12, fill=color or THEME.accent)
    # gắn animation move theo curve bằng background layer đơn giản: bỏ qua, đủ đẹp
    scene.wait(0.6)
    return scene


# --------------------------------------------------------------- neural net
def neural_net_scene(
    title: str,
    layers: Sequence[int] = (3, 5, 4, 2),
    subtitle: str = "",
    duration: Optional[float] = None,
) -> Scene:
    """Mạng neural với tín hiệu chạy từ input -> output."""
    scene = Scene(duration=duration)
    W, H = THEME.width, THEME.height
    _title_block(scene, title, subtitle)
    net = NeuralNet(
        layers=layers,
        box=(W * 0.2, 320, W * 0.8, H - 160),
        node_radius=28,
    )
    net.pulse = 0.0
    scene.play(FadeIn(net, run_time=0.6))
    scene.play(Signal(net, run_time=2.4))
    scene.wait(0.5)
    return scene


# --------------------------------------------------------------- bar chart
def bar_chart_scene(
    title: str,
    values: Sequence[float],
    labels: Optional[Sequence[str]] = None,
    subtitle: str = "",
    duration: Optional[float] = None,
) -> Scene:
    """Biểu đồ cột mọc lên dần (so sánh số liệu)."""
    scene = Scene(duration=duration)
    W, H = THEME.width, THEME.height
    _title_block(scene, title, subtitle)
    chart = BarChart(
        values=values,
        box=(W * 0.16, 340, W * 0.84, H - 180),
        labels=labels,
    )
    chart.progress = 0.0
    scene.play(FadeIn(chart, run_time=0.4))
    scene.play(DrawLine(chart, run_time=1.8))  # DrawLine chỉnh .progress
    scene.wait(0.6)
    return scene


# --------------------------------------------------------------- sorting bars
def sorting_scene(
    title: str,
    data: Optional[Sequence[float]] = None,
    subtitle: str = "",
    duration: Optional[float] = None,
) -> Scene:
    """Minh hoạ sắp xếp (bubble): các bước cột đổi vị trí dần.

    Ta không animate hoán đổi phức tạp; thay vào đó dựng chuỗi trạng thái
    (frame keys) bằng nhiều BarChart chồng thời gian đơn giản: hiển thị data
    ban đầu rồi data đã sort, kèm nhấn mạnh. Đủ trực quan cho video ngắn.
    """
    import random

    scene = Scene(duration=duration)
    W, H = THEME.width, THEME.height
    _title_block(scene, title, subtitle)
    if data is None:
        data = [random.randint(2, 10) for _ in range(8)]
    data = list(data)
    box = (W * 0.18, 340, W * 0.82, H - 180)

    # dựng các bước bubble sort để tạo cảm giác chuyển động
    steps = [list(data)]
    arr = list(data)
    for i in range(len(arr)):
        for j in range(len(arr) - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                steps.append(list(arr))
    # giới hạn số bước để không quá dài
    if len(steps) > 12:
        idx = [int(round(k)) for k in
               [k * (len(steps) - 1) / 11 for k in range(12)]]
        steps = [steps[i] for i in idx]

    charts = []
    for st in steps:
        c = BarChart(values=st, box=box)
        c.opacity = 0.0
        charts.append(c)
        scene.add(c)

    # hiện lần lượt từng bước rồi ẩn bước trước
    per = max(0.25, (scene.duration or (len(steps) * 0.5)) / max(1, len(steps)) if duration else 0.5)
    for k, c in enumerate(charts):
        scene.play(FadeIn(c, run_time=min(0.35, per)))
        scene.wait(max(0.05, per - 0.35))
        if k < len(charts) - 1:
            from .anims import FadeOut

            scene.play(FadeOut(c, run_time=0.15))
    scene.wait(0.4)
    return scene


# --------------------------------------------------------------- number / stat
def counter_scene(
    title: str,
    to_value: float,
    from_value: float = 0.0,
    unit: str = "",
    fmt: str = "{:.0f}",
    subtitle: str = "",
    duration: Optional[float] = None,
) -> Scene:
    """Con số lớn đếm lên (dùng cho thống kê ấn tượng)."""
    scene = Scene(duration=duration)
    W, H = THEME.width, THEME.height
    _title_block(scene, title, subtitle)
    num = Text(fmt.format(from_value), (W / 2, H / 2 + 20), size=200, color=THEME.accent, anchor="mm")
    num.opacity = 0.0
    scene.add(num)
    scene.play(FadeIn(num, run_time=0.4))
    scene.play(CountUp(num, from_value, to_value, fmt=fmt, run_time=2.2))
    if unit:
        u = Text(unit, (W / 2, H / 2 + 190), size=48, color=THEME.muted, anchor="mm", bold=False)
        scene.play(FadeIn(u, run_time=0.5))
    scene.wait(0.6)
    return scene


# --------------------------------------------------------------- concept steps
def steps_scene(
    title: str,
    steps: Sequence[str],
    subtitle: str = "",
    duration: Optional[float] = None,
) -> Scene:
    """Chuỗi bước nối bằng mũi tên, hiện dần từng bước (giải thích quy trình)."""
    from .objects import Rect, Arrow

    scene = Scene(duration=duration)
    W, H = THEME.width, THEME.height
    _title_block(scene, title, subtitle)
    top = 260 if subtitle else 220
    # Bỏ các bước rỗng/khoảng trắng -> tránh vẽ ô trống không có chữ (chồng ô đen).
    steps = [str(s).strip() for s in steps if str(s).strip()]
    n = len(steps)
    if n == 0:
        return scene
    box_h = min(110, (H - top - 120) / n - 20)
    prev_center = None
    for i, s in enumerate(steps):
        y0 = top + i * (box_h + 40)
        box = (W * 0.25, y0, W * 0.75, y0 + box_h)
        col = THEME.color(i)
        rect = Rect(box, fill=THEME.panel, outline=col, width=3, radius=16)
        label = Text(s, ((box[0] + box[2]) / 2 + 20, (y0 + y0 + box_h) / 2), size=32,
                     color=THEME.text, anchor="mm")
        badge = Circle((box[0] + 50, (y0 + y0 + box_h) / 2), 22, fill=col)
        badge_num = Text(str(i + 1), (box[0] + 50, (y0 + y0 + box_h) / 2), size=22, color="#000000", bold=True, anchor="mm")
        g = Group(rect, badge, badge_num, label)
        scene.add(g)
        scene.play(GrowFromCenter(g, run_time=0.5))
        cur_center = ((box[0] + box[2]) / 2, y0)
        if prev_center is not None:
            arr = Arrow(prev_center, (cur_center[0], y0 - 6),
                        color=THEME.muted, width=4)
            arr.progress = 0.0
            scene.add(arr)
            scene.play(DrawLine(arr, run_time=0.3))
        prev_center = ((box[0] + box[2]) / 2, y0 + box_h)
    scene.wait(0.5)
    return scene


# --------------------------------------------------------------- code / terminal
def terminal_scene(
    title: str,
    code_lines: Sequence[str],
    language: str = "bash",
    subtitle: str = "",
    duration: Optional[float] = None,
) -> Scene:
    """Cửa sổ Terminal / IDE macOS với 3 nút màu, hiệu ứng gõ code."""
    from .objects import Rect, Circle, Text
    from .anims import FadeIn, Write, GrowFromCenter

    scene = Scene(duration=duration)
    W, H = THEME.width, THEME.height
    _title_block(scene, title, subtitle)

    top = 230 if subtitle else 190
    win_w = min(1480, int(W * 0.82))
    win_h = min(680, int(H * 0.65))
    x0 = (W - win_w) // 2
    y0 = top
    x1, y1 = x0 + win_w, y0 + win_h
    header_h = 46

    body = Rect((x0, y0, x1, y1), fill="#0d1117", outline="#30363d", width=2, radius=16)
    header = Rect((x0, y0, x1, y0 + header_h), fill="#161b22", outline="#30363d", width=2, radius=16)

    btn_red = Circle((x0 + 28, y0 + header_h / 2), 7, fill="#ff5f56")
    btn_yellow = Circle((x0 + 52, y0 + header_h / 2), 7, fill="#ffbd2e")
    btn_green = Circle((x0 + 76, y0 + header_h / 2), 7, fill="#27c93f")
    tab_title = Text(f"workspace — {language}", (x0 + win_w / 2, y0 + header_h / 2), size=20, color=THEME.muted, anchor="mm")

    win_group = Group(body, header, btn_red, btn_yellow, btn_green, tab_title)
    scene.add(win_group)
    scene.play(GrowFromCenter(win_group, run_time=0.55))

    line_y = y0 + header_h + 36
    lines = [str(l) for l in code_lines if str(l).strip()][:8]
    for i, line in enumerate(lines):
        prompt = "$" if language == "bash" else f"{i + 1:02d}"
        p_col = THEME.accent if language == "bash" else THEME.muted
        p_txt = Text(prompt, (x0 + 36, line_y), size=24, color=p_col, anchor="lm")
        code_txt = Text(line, (x0 + 80, line_y), size=24, color=THEME.text, anchor="lm")
        scene.add(p_txt, code_txt)
        scene.play(FadeIn(p_txt, run_time=0.15), Write(code_txt, run_time=0.45))
        line_y += 42
    scene.wait(0.6)
    return scene


# --------------------------------------------------------------- architecture flow
def architecture_flow_scene(
    title: str,
    nodes: Sequence[str] = ("Client", "API Gateway", "Microservice", "Database"),
    subtitle: str = "",
    duration: Optional[float] = None,
) -> Scene:
    """Sơ đồ kiến trúc luồng dữ liệu các node với tín hiệu chạy qua."""
    from .objects import Rect, Arrow, Circle, Text
    from .anims import FadeIn, FadeOut, DrawLine, Move, GrowFromCenter

    scene = Scene(duration=duration)
    W, H = THEME.width, THEME.height
    _title_block(scene, title, subtitle)

    clean_nodes = [str(n).strip() for n in nodes if str(n).strip()][:5]
    if not clean_nodes:
        clean_nodes = ["Client", "Gateway", "Service", "Database"]

    n = len(clean_nodes)
    box_w = min(250, int((W * 0.82) / n - 36))
    box_h = 120
    gap = (W * 0.82 - n * box_w) / max(1, n - 1)
    start_x = (W - (n * box_w + (n - 1) * gap)) / 2
    cy = H * 0.54

    centers = []
    for i, name in enumerate(clean_nodes):
        bx0 = start_x + i * (box_w + gap)
        bx1 = bx0 + box_w
        by0 = cy - box_h / 2
        by1 = cy + box_h / 2
        col = THEME.color(i)
        r = Rect((bx0, by0, bx1, by1), fill="#161b22", outline=col, width=3, radius=18)
        lbl = Text(name, ((bx0 + bx1) / 2, cy), size=24, color=THEME.text, anchor="mm")
        g = Group(r, lbl)
        scene.add(g)
        scene.play(GrowFromCenter(g, run_time=0.4))
        centers.append(((bx0 + bx1) / 2, cy))

    arrows = []
    for i in range(n - 1):
        p1 = (centers[i][0] + box_w / 2 + 4, cy)
        p2 = (centers[i + 1][0] - box_w / 2 - 4, cy)
        arr = Arrow(p1, p2, color=THEME.accent, width=4)
        scene.add(arr)
        scene.play(DrawLine(arr, run_time=0.25))
        arrows.append((p1, p2))

    for i in range(n - 1):
        p1, p2 = arrows[i]
        packet = Circle(p1, 9, fill="#39ff14")
        scene.add(packet)
        scene.play(Move(packet, dx=p2[0] - p1[0], dy=0, run_time=0.45, easing="ease_in_out"))
        scene.play(FadeOut(packet, run_time=0.15))

    scene.wait(0.6)
    return scene


# --------------------------------------------------------------- comparison (2 columns)
def comparison_scene(
    title: str,
    left_title: str = "Cách ngây thơ",
    left_items: Sequence[str] = ("Chậm O(N^2)", "Dễ tràn bộ nhớ", "Bế tắc khi dữ liệu lớn"),
    right_title: str = "Giải pháp tối ưu",
    right_items: Sequence[str] = ("Nhanh O(N log N)", "Tiết kiệm RAM", "Đáp ứng triệu QPS"),
    subtitle: str = "",
    duration: Optional[float] = None,
) -> Scene:
    """So sánh 2 giải pháp trực quan 2 cột (Naive vs Optimal)."""
    from .objects import Rect, Text
    from .anims import FadeIn, GrowFromCenter

    scene = Scene(duration=duration)
    W, H = THEME.width, THEME.height
    _title_block(scene, title, subtitle)

    card_w = min(640, int(W * 0.38))
    card_h = min(520, int(H * 0.52))
    cy = H * 0.55
    y0, y1 = cy - card_h / 2, cy + card_h / 2

    # Cột trái (Naive)
    lx0 = W * 0.50 - card_w - 24
    lx1 = lx0 + card_w
    l_bg = Rect((lx0, y0, lx1, y1), fill="#161b22", outline="#ff7b72", width=3, radius=18)
    l_hdr = Rect((lx0, y0, lx1, y0 + 60), fill="#ff7b72", outline="#ff7b72", width=1, radius=18)
    l_title = Text("✕  " + left_title, ((lx0 + lx1) / 2, y0 + 30), size=26, color="#000000", bold=True, anchor="mm")
    l_group = Group(l_bg, l_hdr, l_title)
    scene.add(l_group)
    scene.play(GrowFromCenter(l_group, run_time=0.45))

    ly = y0 + 96
    for item in list(left_items)[:4]:
        it = Text("• " + str(item), (lx0 + 36, ly), size=24, color="#f0883e", anchor="lm")
        scene.add(it)
        scene.play(FadeIn(it, shift=10, run_time=0.2))
        ly += 50

    # Cột phải (Optimal)
    rx0 = W * 0.50 + 24
    rx1 = rx0 + card_w
    r_bg = Rect((rx0, y0, rx1, y1), fill="#161b22", outline="#39ff14", width=3, radius=18)
    r_hdr = Rect((rx0, y0, rx1, y0 + 60), fill="#238636", outline="#39ff14", width=1, radius=18)
    r_title = Text("✓  " + right_title, ((rx0 + rx1) / 2, y0 + 30), size=26, color="#ffffff", bold=True, anchor="mm")
    r_group = Group(r_bg, r_hdr, r_title)
    scene.add(r_group)
    scene.play(GrowFromCenter(r_group, run_time=0.45))

    ry = y0 + 96
    for item in list(right_items)[:4]:
        it = Text("• " + str(item), (rx0 + 36, ry), size=24, color="#3fb950", anchor="lm")
        scene.add(it)
        scene.play(FadeIn(it, shift=10, run_time=0.2))
        ry += 50

    scene.wait(0.6)
    return scene


# --------------------------------------------------------------- mascot (Bunny Girl / Stickman)
def mascot_scene(
    title: str,
    mascot: str = "bunny_girl",
    bubble_title: str = "Lời khuyên & Best Practice",
    tips: Sequence[str] = ("Chia nhỏ bài toán phức tạp", "Viết code sạch và dễ bảo trì", "Tối ưu hóa hiệu năng và bộ nhớ"),
    subtitle: str = "",
    duration: Optional[float] = None,
) -> Scene:
    """Scene linh vật (Bunny Girl hoặc Stickman) tương tác giải thích tech / best practice."""
    from .objects import Mascot, Rect, Text
    from .anims import FadeIn, GrowFromCenter

    scene = Scene(duration=duration)
    W, H = THEME.width, THEME.height
    _title_block(scene, title, subtitle)

    # 1. Mascot bên trái
    mx = W * 0.22
    my = H * 0.58
    m = Mascot((mx, my), kind=mascot, size=150)
    scene.add(m)
    scene.play(GrowFromCenter(m, run_time=0.55))

    # 2. Khung thoại bên phải
    bx0 = W * 0.42
    bx1 = W * 0.90
    by0 = H * 0.35
    by1 = H * 0.85
    b_bg = Rect((bx0, by0, bx1, by1), fill="#161b22", outline="#fbbf24", width=3, radius=20)
    b_hdr = Rect((bx0, by0, bx1, by0 + 64), fill="#fbbf24", outline="#fbbf24", width=1, radius=20)
    b_title = Text("★  " + bubble_title, ((bx0 + bx1) / 2, by0 + 32), size=28, color="#000000", bold=True, anchor="mm")
    b_group = Group(b_bg, b_hdr, b_title)
    scene.add(b_group)
    scene.play(GrowFromCenter(b_group, run_time=0.45))

    ty = by0 + 104
    for tip in list(tips)[:4]:
        it = Text("✔  " + str(tip), (bx0 + 36, ty), size=24, color="#ffffff", anchor="lm")
        scene.add(it)
        scene.play(FadeIn(it, shift=12, run_time=0.25))
        ty += 56

    scene.wait(0.6)
    return scene


def math_equation_scene(
    title: str,
    formula: str,
    parts: list[dict] | None = None,
    subtitle: str = "",
    duration: float = 6.0,
) -> Scene:
    """Scene trình bày công thức toán học sắc nét với bóc tách thành phần (3Blue1Brown)."""
    from .anims import FadeIn, GrowFromCenter
    from .objects import Formula, Group, Rect, Text

    scene = Scene(duration=duration)
    W, H = THEME.width, THEME.height
    _title_block(scene, title, subtitle)

    # 1. Khung công thức chính ở trên
    fx0, fx1 = W * 0.1, W * 0.9
    fy0, fy1 = H * 0.28, H * 0.48
    f_bg = Rect((fx0, fy0, fx1, fy1), fill="#0d1117", outline="#fbbf24", width=3, radius=16)
    f_obj = Formula(formula, ((fx0 + fx1) / 2, (fy0 + fy1) / 2), size=55, color="#ffffff", anchor="mm")
    f_group = Group(f_bg, f_obj)
    scene.add(f_group)
    scene.play(GrowFromCenter(f_group, run_time=0.6))

    # 2. Các thẻ bóc tách thành phần ở dưới
    p_list = parts or [
        {"symbol": "QK^T", "name": "Attention Score", "desc": "Tích vô hướng", "color": "#00f0ff"},
        {"symbol": "\\sqrt{d_k}", "name": "Scaling Factor", "desc": "Ổn định gradient", "color": "#fbbf24"},
        {"symbol": "V", "name": "Value Vector", "desc": "Tổng hợp ngữ nghĩa", "color": "#38ef7d"},
    ]
    n_p = min(len(p_list), 4)
    if n_p > 0:
        gap = 20
        cw = (fx1 - fx0 - (n_p - 1) * gap) / n_p
        cy0, cy1 = H * 0.53, H * 0.85
        for i, p in enumerate(p_list[:n_p]):
            cx = fx0 + i * (cw + gap)
            col = p.get("color") or "#00f0ff"
            p_bg = Rect((cx, cy0, cx + cw, cy1), fill="#161b22", outline=col, width=2, radius=12)
            sym = Formula(str(p.get("symbol", "")), (cx + cw / 2, cy0 + 45), size=32, color=col, anchor="mm")
            nm = Text(str(p.get("name", "")), (cx + cw / 2, cy0 + 95), size=19, color="#ffffff", bold=True, anchor="mm")
            desc = Text(str(p.get("desc", "")), (cx + cw / 2, cy0 + 130), size=15, color="#94a3b8", anchor="mm")
            p_card = Group(p_bg, sym, nm, desc)
            scene.add(p_card)
            scene.play(FadeIn(p_card, shift=15, run_time=0.3))

    scene.wait(0.6)
    return scene


# --------------------------------------------------------------- stickman explainer
def stickman_scene(
    title: str,
    pose: str = "pointing",
    bullets: Optional[Sequence[str]] = None,
    code_snippet: str = "",
    subtitle: str = "",
    duration: float = 6.0,
) -> Scene:
    """Scene người que giải thích kiến thức (StykAnimate Stick Figure Explainer).

    Người que đứng phía bên trái dùng khớp IK chỉ que laser deictic vào thẻ kiến thức bên phải.
    """
    W, H = THEME.width, THEME.height
    scene = Scene(duration=duration)
    _title_block(scene, title, subtitle)

    # Bên trái: Người que StykAnimate
    stick_x = W * 0.20
    stick_y = H * 0.60
    card_x0 = W * 0.38
    card_x1 = W * 0.90
    card_y0 = H * 0.30
    card_y1 = H * 0.86

    target_pt = (card_x0 + 40, card_y0 + 60)
    stick = Stickman(center=(stick_x, stick_y), pose=pose, scale=1.9, color=THEME.accent, target_point=target_pt)
    scene.add(stick)
    scene.play(FadeIn(stick, shift=20, run_time=0.6))

    # Bên phải: Thẻ kiến thức / Bảng giải thích
    panel = Rect((card_x0, card_y0, card_x1, card_y1), fill="#0d1117", outline=THEME.secondary, width=2.5, radius=16)
    scene.add(panel)
    scene.play(FadeIn(panel, run_time=0.4))

    # Tiêu đề bảng
    header_bar = Rect((card_x0, card_y0, card_x1, card_y0 + 60), fill="#161b22", outline=THEME.secondary, width=1.5, radius=16)
    badge = Text("TRỌNG TÂM BẢN CHẤT", (card_x0 + 30, card_y0 + 30), size=24, color=THEME.secondary, anchor="lm")
    scene.add(header_bar)
    scene.add(badge)

    # Nội dung bullets
    lines = bullets or ["Trực quan hoá bản chất cơ chế", "Giải thích từng bước không cưỡi ngựa xem hoa", "Tối ưu hiệu năng và độ phức tạp"]
    row_h = (card_y1 - card_y0 - 90) / max(len(lines), 1)
    for i, line_txt in enumerate(lines[:4]):
        ly = card_y0 + 80 + i * row_h + row_h / 2
        dot = Circle((card_x0 + 36, ly), 5, fill=THEME.accent)
        txt = Text(str(line_txt)[:55], (card_x0 + 58, ly), size=26, color=THEME.text, anchor="lm", bold=False)
        row = Group(dot, txt)
        scene.add(row)
        scene.play(FadeIn(row, shift=15, run_time=0.4))

    scene.wait(0.8)
    return scene


# preset registry để pipeline chọn theo tên
PRESETS = {
    "function": function_scene,
    "neural_net": neural_net_scene,
    "bar_chart": bar_chart_scene,
    "sorting": sorting_scene,
    "counter": counter_scene,
    "steps": steps_scene,
    "terminal": terminal_scene,
    "code": terminal_scene,
    "architecture_flow": architecture_flow_scene,
    "architecture": architecture_flow_scene,
    "comparison": comparison_scene,
    "mascot": mascot_scene,
    "mascot_tip": mascot_scene,
    "math_equation": math_equation_scene,
    "formula": math_equation_scene,
    "equation": math_equation_scene,
    "stickman": stickman_scene,
    "stickman_explainer": stickman_scene,
}



