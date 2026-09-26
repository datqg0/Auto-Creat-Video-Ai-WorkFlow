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
        rect = Rect(box, fill=THEME.panel, outline=THEME.color(i), width=3, radius=16)
        rect.opacity = 0.0
        label = Text(s, ((box[0] + box[2]) / 2, (y0 + y0 + box_h) / 2), size=34,
                     color=THEME.text, anchor="mm")
        label.opacity = 0.0
        g = Group(rect, label)
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


# preset registry để pipeline chọn theo tên
PRESETS = {
    "function": function_scene,
    "neural_net": neural_net_scene,
    "bar_chart": bar_chart_scene,
    "sorting": sorting_scene,
    "counter": counter_scene,
    "steps": steps_scene,
}
