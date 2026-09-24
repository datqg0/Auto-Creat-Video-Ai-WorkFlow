"""Trình dựng animation TUỲ BIẾN từ 1 "spec" khai báo (JSON) do LLM viết.

Mục tiêu: cho phép AI tự "viết code" animation minh hoạ đa dạng hơn 6 preset
cố định, GIỐNG cách ChatGPT viết code minh hoạ — nhưng KHÔNG chạy code Python
tuỳ ý của LLM (không eval/exec). Thay vào đó LLM mô tả cảnh bằng 1 danh sách
đối tượng + timeline animation; ta DIỄN GIẢI an toàn sang primitives mathviz.

Định dạng spec (scene.animation khi preset == "custom"):
    {
      "preset": "custom",
      "objects": [
        {"id": "t1", "type": "text", "text": "E = mc^2", "x": "0.5W", "y": 160,
         "size": 72, "color": "accent", "bold": true, "anchor": "mm"},
        {"id": "ax", "type": "axes", "x0": "0.14W", "y0": 320, "x1": "0.86W",
         "y1": "0.85H", "x_range": [-6.28, 6.28], "y_range": [-1.4, 1.4]},
        {"id": "g",  "type": "graph", "axes": "ax", "expr": "sin(x)", "color": "#3fb950"},
        {"id": "box","type": "rect", "x0": "0.3W", "y0": 500, "x1": "0.7W",
         "y1": 620, "fill": "panel", "outline": "accent", "radius": 16},
        {"id": "ar", "type": "arrow", "x1": "0.5W", "y1": 640, "x2": "0.5W", "y2": 760}
      ],
      "timeline": [
        {"play": [{"anim": "write", "target": "t1", "run_time": 1.0}]},
        {"play": [{"anim": "fade_in", "target": "ax", "run_time": 0.6}]},
        {"play": [{"anim": "draw", "target": "g", "run_time": 2.0}]},
        {"wait": 0.5},
        {"play": [
           {"anim": "grow", "target": "box", "run_time": 0.6},
           {"anim": "draw", "target": "ar", "run_time": 0.4}
        ]}
      ]
    }

Loại đối tượng: text, dot, line, arrow, rect, circle, axes, graph, neural_net,
bar_chart.
Loại animation: fade_in, fade_out, write, draw, grow, move, count_up, pulse,
signal.

Toạ độ: số px tuyệt đối, HOẶC chuỗi dạng "0.5W" / "0.85H" (phần trăm khung).
Màu: hex "#rrggbb" HOẶC tên theme (accent/text/muted/panel/grid) HOẶC "c0".."c6".
Biểu thức đồ thị: chỉ qua make_safe_fn (chỉ toán học, không exec).

An toàn: giới hạn số đối tượng/bước, clamp kích thước, KHÔNG có eval/exec.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

from .core import Scene
from .objects import (
    Arrow,
    Axes,
    BarChart,
    Circle,
    Dot,
    FunctionGraph,
    Line,
    NeuralNet,
    Rect,
    Text,
)
from .anims import (
    CountUp,
    DrawLine,
    FadeIn,
    FadeOut,
    GrowFromCenter,
    Move,
    Pulse,
    Signal,
    Write,
)
from .theme import THEME

log = logging.getLogger(__name__)

# --- giới hạn an toàn (tránh spec quá lớn làm render treo) ---
_MAX_OBJECTS = 40
_MAX_STEPS = 80
_MAX_ANIMS_PER_STEP = 8
_MAX_RUN_TIME = 6.0
_MAX_LAYER_NODES = 12
_MAX_LAYERS = 8
_MAX_BARS = 16

_HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
_THEME_COLORS = {
    "accent": lambda: THEME.accent,
    "text": lambda: THEME.text,
    "muted": lambda: THEME.muted,
    "panel": lambda: THEME.panel,
    "grid": lambda: THEME.grid,
    "bg": lambda: THEME.bg,
}


def _num(v: Any, default: float = 0.0) -> float:
    """Ép về float an toàn."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _coord(v: Any, axis: str) -> float:
    """Chuyển toạ độ: số -> px; chuỗi '0.5W'/'0.85H' -> phần trăm khung.

    ``axis`` chỉ dùng khi giá trị là số trần và ta cần fallback theo trục
    (không thực sự cần, giữ chữ ký gọn). Trả về px tuyệt đối, clamp trong khung.
    """
    W, H = THEME.width, THEME.height
    if isinstance(v, str):
        s = v.strip()
        m = re.match(r"^([0-9]*\.?[0-9]+)\s*([WwHh])$", s)
        if m:
            frac = float(m.group(1))
            base = W if m.group(2).lower() == "w" else H
            val = frac * base
        else:
            val = _num(s, 0.0)
    else:
        val = _num(v, 0.0)
    lim = W if axis == "x" else H
    # cho phép hơi tràn nhẹ nhưng chặn giá trị vô lý
    return max(-lim, min(lim * 2, val))


def _color(v: Any, default: Optional[str] = None) -> Optional[str]:
    """Chuẩn hoá màu: hex hợp lệ / tên theme / 'c<idx>'. None -> mặc định."""
    if v is None:
        return default
    if isinstance(v, str):
        s = v.strip()
        if _HEX_RE.match(s):
            return s
        low = s.lower()
        if low in _THEME_COLORS:
            return _THEME_COLORS[low]()
        m = re.match(r"^c(\d+)$", low)
        if m:
            return THEME.color(int(m.group(1)))
    return default


def _bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y")
    return default


def _range2(v: Any, default: tuple[float, float]) -> tuple[float, float]:
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        return (_num(v[0], default[0]), _num(v[1], default[1]))
    return default


def _build_object(spec: dict, safe_fn_maker, objs: dict):
    """Dựng 1 Drawable từ 1 mô tả đối tượng. Trả về None nếu type lạ."""
    typ = str(spec.get("type", "")).strip().lower()
    if typ == "text":
        return Text(
            str(spec.get("text", "")),
            (_coord(spec.get("x", "0.5W"), "x"), _coord(spec.get("y", 200), "y")),
            size=int(max(12, min(240, _num(spec.get("size", 48), 48)))),
            color=_color(spec.get("color"), THEME.text),
            bold=_bool(spec.get("bold", True), True),
            anchor=str(spec.get("anchor", "mm")),
        )
    if typ == "dot":
        return Dot(
            (_coord(spec.get("x", "0.5W"), "x"), _coord(spec.get("y", "0.5H"), "y")),
            radius=max(2, min(80, _num(spec.get("radius", 10), 10))),
            color=_color(spec.get("color"), THEME.accent),
        )
    if typ in ("line", "arrow"):
        cls = Arrow if typ == "arrow" else Line
        return cls(
            (_coord(spec.get("x1", 0), "x"), _coord(spec.get("y1", 0), "y")),
            (_coord(spec.get("x2", 100), "x"), _coord(spec.get("y2", 0), "y")),
            color=_color(spec.get("color"), THEME.text),
            width=max(1, min(20, _num(spec.get("width", 4), 4))),
        )
    if typ == "rect":
        return Rect(
            (
                _coord(spec.get("x0", "0.3W"), "x"),
                _coord(spec.get("y0", 400), "y"),
                _coord(spec.get("x1", "0.7W"), "x"),
                _coord(spec.get("y1", 560), "y"),
            ),
            fill=_color(spec.get("fill"), None),
            outline=_color(spec.get("outline"), None),
            width=max(1, min(20, _num(spec.get("width", 3), 3))),
            radius=max(0, min(60, _num(spec.get("radius", 14), 14))),
        )
    if typ == "circle":
        return Circle(
            (_coord(spec.get("x", "0.5W"), "x"), _coord(spec.get("y", "0.5H"), "y")),
            radius=max(2, min(400, _num(spec.get("radius", 60), 60))),
            fill=_color(spec.get("fill"), None),
            outline=_color(spec.get("outline"), THEME.accent),
            width=max(1, min(20, _num(spec.get("width", 3), 3))),
        )
    if typ == "axes":
        return Axes(
            box=(
                _coord(spec.get("x0", "0.14W"), "x"),
                _coord(spec.get("y0", 320), "y"),
                _coord(spec.get("x1", "0.86W"), "x"),
                _coord(spec.get("y1", "0.85H"), "y"),
            ),
            x_range=_range2(spec.get("x_range"), (-6.28, 6.28)),
            y_range=_range2(spec.get("y_range"), (-1.4, 1.4)),
            grid=_bool(spec.get("grid", True), True),
        )
    if typ == "graph":
        ax = objs.get(str(spec.get("axes", "")))
        if not isinstance(ax, Axes):
            log.warning("graph thiếu axes hợp lệ: %r", spec.get("axes"))
            return None
        fn = safe_fn_maker(str(spec.get("expr", "sin(x)")))
        return FunctionGraph(
            ax,
            fn,
            color=_color(spec.get("color"), THEME.accent),
            width=max(1, min(14, _num(spec.get("width", 5), 5))),
        )
    if typ == "neural_net":
        layers = spec.get("layers", [3, 5, 4, 2])
        if not isinstance(layers, (list, tuple)) or not layers:
            layers = [3, 5, 4, 2]
        layers = [int(max(1, min(_MAX_LAYER_NODES, _num(n, 1)))) for n in layers][:_MAX_LAYERS]
        return NeuralNet(
            layers=layers,
            box=(
                _coord(spec.get("x0", "0.2W"), "x"),
                _coord(spec.get("y0", 320), "y"),
                _coord(spec.get("x1", "0.8W"), "x"),
                _coord(spec.get("y1", "0.85H"), "y"),
            ),
            node_radius=max(8, min(48, _num(spec.get("node_radius", 26), 26))),
        )
    if typ == "bar_chart":
        values = spec.get("values", [3, 6, 4, 8])
        if not isinstance(values, (list, tuple)) or not values:
            values = [3, 6, 4, 8]
        values = [_num(v, 0) for v in values][:_MAX_BARS]
        labels = spec.get("labels")
        if isinstance(labels, (list, tuple)):
            labels = [str(x) for x in labels][:_MAX_BARS]
        else:
            labels = None
        return BarChart(
            values=values,
            box=(
                _coord(spec.get("x0", "0.16W"), "x"),
                _coord(spec.get("y0", 340), "y"),
                _coord(spec.get("x1", "0.84W"), "x"),
                _coord(spec.get("y1", "0.85H"), "y"),
            ),
            labels=labels,
        )
    log.warning("Bỏ đối tượng type lạ: %r", typ)
    return None


# tên anim -> hàm dựng Animation(target, **kwargs an toàn)
def _make_anim(name: str, target, spec: dict):
    rt = max(0.05, min(_MAX_RUN_TIME, _num(spec.get("run_time", 1.0), 1.0)))
    name = name.strip().lower()
    if name == "fade_in":
        return FadeIn(target, run_time=rt, shift=_num(spec.get("shift", 0), 0))
    if name == "fade_out":
        return FadeOut(target, run_time=rt, shift=_num(spec.get("shift", 0), 0))
    if name == "write":
        return Write(target, run_time=rt)
    if name == "draw":
        return DrawLine(target, run_time=rt)
    if name == "grow":
        return GrowFromCenter(target, run_time=rt)
    if name == "move":
        return Move(
            target,
            dx=_num(spec.get("dx", 0), 0),
            dy=_num(spec.get("dy", 0), 0),
            run_time=rt,
        )
    if name == "count_up":
        return CountUp(
            target,
            from_value=_num(spec.get("from", 0), 0),
            to_value=_num(spec.get("to", 100), 100),
            run_time=rt,
            fmt=str(spec.get("fmt", "{:.0f}")),
        )
    if name == "pulse":
        return Pulse(
            target,
            amount=max(0.0, min(1.0, _num(spec.get("amount", 0.15), 0.15))),
            cycles=max(0.5, min(6, _num(spec.get("cycles", 1), 1))),
            run_time=rt,
        )
    if name == "signal":
        return Signal(target, run_time=rt)
    log.warning("Bỏ animation lạ: %r", name)
    return None


def build_custom_scene(
    spec: dict,
    safe_fn_maker,
    title: str = "",
    subtitle: str = "",
    duration: Optional[float] = None,
) -> Scene:
    """Dựng Scene từ spec khai báo. ``safe_fn_maker`` = make_safe_fn (bơm vào để
    tránh import vòng)."""
    scene = Scene(duration=duration)
    W = THEME.width

    # tiêu đề trên cùng (tuỳ chọn) — dùng chung phong cách với các preset khác
    if title:
        t = Text(title, (W / 2, 120), size=64, color=THEME.text, anchor="mm")
        scene.play(FadeIn(t, shift=30, run_time=0.6))
        if subtitle:
            s = Text(subtitle, (W / 2, 200), size=34, color=THEME.muted,
                     anchor="mm", bold=False)
            scene.play(FadeIn(s, run_time=0.4))

    # 1) dựng toàn bộ đối tượng theo id
    objs: dict[str, Any] = {}
    raw_objs = spec.get("objects", [])
    if not isinstance(raw_objs, list):
        raw_objs = []
    for od in raw_objs[:_MAX_OBJECTS]:
        if not isinstance(od, dict):
            continue
        oid = str(od.get("id", "")).strip()
        if not oid:
            continue
        obj = _build_object(od, safe_fn_maker, objs)
        if obj is not None:
            objs[oid] = obj

    if not objs:
        log.warning("custom scene: không có đối tượng hợp lệ")
        scene.wait(duration or 1.0)
        return scene

    # 2) chạy timeline
    timeline = spec.get("timeline", [])
    if not isinstance(timeline, list):
        timeline = []
    played_any = False
    for step in timeline[:_MAX_STEPS]:
        if not isinstance(step, dict):
            continue
        if "wait" in step:
            scene.wait(max(0.0, min(_MAX_RUN_TIME, _num(step.get("wait", 0.5), 0.5))))
            continue
        if "add" in step:
            ids = step.get("add", [])
            if isinstance(ids, (list, tuple)):
                to_add = [objs[str(i)] for i in ids if str(i) in objs]
                if to_add:
                    scene.add(*to_add)
                    played_any = True
            continue
        if "play" in step:
            raw_anims = step.get("play", [])
            if not isinstance(raw_anims, list):
                continue
            anims = []
            for ad in raw_anims[:_MAX_ANIMS_PER_STEP]:
                if not isinstance(ad, dict):
                    continue
                target = objs.get(str(ad.get("target", "")))
                if target is None:
                    continue
                a = _make_anim(str(ad.get("anim", "")), target, ad)
                if a is not None:
                    anims.append(a)
            if anims:
                scene.play(*anims)
                played_any = True

    # nếu LLM quên timeline: hiện dần mọi đối tượng cho khỏi trống
    if not played_any:
        for obj in objs.values():
            scene.play(FadeIn(obj, run_time=0.4))
    scene.wait(0.4)
    return scene
