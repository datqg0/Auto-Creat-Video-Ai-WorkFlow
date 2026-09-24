"""Cầu nối giữa model Scene (pipeline) và thư viện mathviz.

``build_animation_scene(scene, duration)`` đọc ``scene.animation`` (dict) rồi
gọi preset tương ứng trong mathviz.scenes, khớp thời lượng với audio narration.

Cấu hình animation trong model:
    {
      "preset": "function" | "neural_net" | "bar_chart" | "sorting" | "counter" | "steps",
      ... tham số tuỳ preset ...
    }

Ví dụ:
    {"preset": "function", "expr": "sin(x)", "x_range": [-6.28, 6.28]}
    {"preset": "neural_net", "layers": [3, 5, 4, 2]}
    {"preset": "bar_chart", "values": [3,7,5], "labels": ["A","B","C"]}
    {"preset": "counter", "to_value": 1000000, "unit": "người dùng"}
    {"preset": "steps", "steps": ["Bước 1", "Bước 2", "Bước 3"]}
    {"preset": "sorting", "data": [5,2,8,1,9]}
"""
from __future__ import annotations

import ast
import logging
import math
import operator
from typing import Callable, Optional

from .models import Scene as ModelScene

log = logging.getLogger(__name__)

# --- an toàn: eval biểu thức toán học từ LLM mà không dùng eval() thô ---
_ALLOWED_FUNCS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "log": math.log,
    "sqrt": lambda v: math.sqrt(abs(v)),
    "abs": abs,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "floor": math.floor,
    "ceil": math.ceil,
}
_ALLOWED_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def make_safe_fn(expr: str) -> Callable[[float], float]:
    """Biến chuỗi 'sin(x)*x' thành hàm f(x) an toàn (chỉ toán học, không exec tuỳ ý)."""
    tree = ast.parse(expr, mode="eval").body

    def _eval(node, x: float) -> float:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError("hằng không hợp lệ")
        if isinstance(node, ast.Name):
            if node.id == "x":
                return x
            if node.id in _ALLOWED_CONSTS:
                return _ALLOWED_CONSTS[node.id]
            raise ValueError(f"tên không cho phép: {node.id}")
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](_eval(node.left, x), _eval(node.right, x))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_eval(node.operand, x))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = _ALLOWED_FUNCS.get(node.func.id)
            if fn is None:
                raise ValueError(f"hàm không cho phép: {node.func.id}")
            args = [_eval(a, x) for a in node.args]
            return float(fn(*args))
        raise ValueError("biểu thức không hợp lệ")

    def f(x: float) -> float:
        try:
            return _eval(tree, x)
        except Exception:  # noqa: BLE001
            return float("nan")

    return f


def build_animation_scene(scene: ModelScene, duration: float):
    """Dựng 1 mathviz.Scene từ scene.animation, đã fit đúng duration. None nếu lỗi."""
    from .mathviz import scenes as mv_scenes  # lazy

    cfg = scene.animation or {}
    preset = str(cfg.get("preset", "")).strip()
    title = scene.heading or cfg.get("title", "")
    subtitle = cfg.get("subtitle", "")

    try:
        if preset == "function":
            expr = cfg.get("expr", "sin(x)")
            fn = make_safe_fn(expr)
            xr = tuple(cfg.get("x_range", (-6.28, 6.28)))
            yr = tuple(cfg.get("y_range", (-1.4, 1.4)))
            sc = mv_scenes.function_scene(
                title or f"y = {expr}", fn, x_range=xr, y_range=yr, subtitle=subtitle
            )
        elif preset == "neural_net":
            layers = cfg.get("layers", [3, 5, 4, 2])
            sc = mv_scenes.neural_net_scene(
                title or "Mạng Neural", layers=layers, subtitle=subtitle
            )
        elif preset == "bar_chart":
            sc = mv_scenes.bar_chart_scene(
                title or "So sánh",
                values=cfg.get("values", [3, 6, 4, 8]),
                labels=cfg.get("labels"),
                subtitle=subtitle,
            )
        elif preset == "sorting":
            sc = mv_scenes.sorting_scene(
                title or "Sắp xếp",
                data=cfg.get("data"),
                subtitle=subtitle,
                duration=duration,
            )
        elif preset == "counter":
            sc = mv_scenes.counter_scene(
                title or "Con số",
                to_value=float(cfg.get("to_value", 100)),
                from_value=float(cfg.get("from_value", 0)),
                unit=cfg.get("unit", ""),
                fmt=cfg.get("fmt", "{:.0f}"),
                subtitle=subtitle,
            )
        elif preset == "steps":
            sc = mv_scenes.steps_scene(
                title or "Quy trình",
                steps=cfg.get("steps", scene.bullets or ["Bước 1", "Bước 2"]),
                subtitle=subtitle,
            )
        elif preset == "custom":
            # Animation TUỲ BIẾN do LLM "viết": mô tả bằng spec khai báo
            # (objects + timeline) -> diễn giải an toàn, KHÔNG exec code.
            from .mathviz.custom_scene import build_custom_scene  # lazy

            sc = build_custom_scene(
                cfg,
                make_safe_fn,
                title=title,
                subtitle=subtitle,
                duration=duration,
            )
        else:
            log.warning("Preset animation không hỗ trợ: %r", preset)
            return None
    except Exception as e:  # noqa: BLE001
        log.warning("Dựng animation lỗi (%s): %s", preset, e)
        return None

    sc.fit_duration(duration)
    return sc
