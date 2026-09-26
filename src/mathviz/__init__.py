"""mathviz — thư viện visual math/tech tự viết, nhẹ và cơ động cao.

Triết lý lấy cảm hứng từ Manim (3Blue1Brown): mô tả cảnh bằng "mảnh" (drawable)
và "animation" theo thời gian, nhưng render bằng Pillow + numpy nên nhanh, không
cần LaTeX/cairo, chạy tốt trên CI.

Cách dùng cơ bản:
    from src.mathviz import Scene
    scene = Scene(duration=6.0)
    scene.add(...)              # thêm drawable tĩnh
    scene.play(FadeIn(obj))     # thêm animation
    scene.render_mp4(out_path)  # xuất video
    scene.render_png(out_path)  # xuất 1 frame (preview)

Toàn bộ toạ độ dùng hệ pixel 1920x1080 (gốc trên-trái) để dễ kiểm soát.
"""
from __future__ import annotations

from .core import Scene
from .anims import (
    FadeIn,
    FadeOut,
    Write,
    DrawLine,
    GrowFromCenter,
    Move,
    MoveAlongPath,
    CountUp,
    Pulse,
    Signal,
    CameraMove,
    Transform,
    MorphShape,
)
from .objects import (
    Text,
    Dot,
    Line,
    Arrow,
    Rect,
    Circle,
    Axes,
    FunctionGraph,
    ParametricCurve,
    Polygon,
    Formula,
    NeuralNet,
    BarChart,
    Group,
)
from .theme import THEME

__all__ = [
    "Scene",
    "FadeIn",
    "FadeOut",
    "Write",
    "DrawLine",
    "GrowFromCenter",
    "Move",
    "MoveAlongPath",
    "CountUp",
    "Pulse",
    "Signal",
    "CameraMove",
    "Transform",
    "MorphShape",
    "Text",
    "Dot",
    "Line",
    "Arrow",
    "Rect",
    "Circle",
    "Axes",
    "FunctionGraph",
    "ParametricCurve",
    "Polygon",
    "Formula",
    "NeuralNet",
    "BarChart",
    "Group",
    "THEME",
]
