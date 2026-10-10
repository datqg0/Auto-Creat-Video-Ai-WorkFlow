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


def make_safe_fn(expr: str, var: str = "x") -> Callable[[float], float]:
    """Biến chuỗi 'sin(x)*x' thành hàm f(v) an toàn (chỉ toán học, không exec tuỳ ý).

    ``var`` = tên biến tự do trong biểu thức (mặc định 'x'; dùng 't' cho đường
    cong tham số). Chỉ đúng 1 biến; mọi tên khác (ngoài hằng pi/e/tau) bị chặn.
    """
    tree = ast.parse(expr, mode="eval").body

    def _eval(node, v: float) -> float:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError("hằng không hợp lệ")
        if isinstance(node, ast.Name):
            if node.id == var:
                return v
            if node.id in _ALLOWED_CONSTS:
                return _ALLOWED_CONSTS[node.id]
            raise ValueError(f"tên không cho phép: {node.id}")
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](_eval(node.left, v), _eval(node.right, v))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_eval(node.operand, v))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = _ALLOWED_FUNCS.get(node.func.id)
            if fn is None:
                raise ValueError(f"hàm không cho phép: {node.func.id}")
            args = [_eval(a, v) for a in node.args]
            return float(fn(*args))
        raise ValueError("biểu thức không hợp lệ")

    def f(v: float) -> float:
        try:
            return _eval(tree, v)
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
        elif preset in ("terminal", "code"):
            lines = cfg.get("code") or cfg.get("lines") or scene.bullets or ["print('Hello, world!')"]
            if isinstance(lines, str):
                lines = lines.splitlines()
            sc = mv_scenes.terminal_scene(
                title or "Thực thi mã lệnh",
                code_lines=lines,
                language=str(cfg.get("language", scene.code_language or "python")),
                subtitle=subtitle,
            )
        elif preset in ("architecture_flow", "architecture"):
            nodes = cfg.get("nodes") or scene.bullets or ["Client", "API Gateway", "Service", "Database"]
            sc = mv_scenes.architecture_flow_scene(
                title or "Kiến trúc hệ thống",
                nodes=nodes,
                subtitle=subtitle,
            )
        elif preset == "comparison":
            sc = mv_scenes.comparison_scene(
                title or "So sánh giải pháp",
                left_title=str(cfg.get("left_title", "Cách ngây thơ")),
                left_items=cfg.get("left_items", ["Chậm O(N^2)", "Dễ quá tải"]),
                right_title=str(cfg.get("right_title", "Giải pháp tối ưu")),
                right_items=cfg.get("right_items", ["Nhanh O(N log N)", "Tối ưu bộ nhớ"]),
                subtitle=subtitle,
            )
        elif preset in ("pycode", "manim", "motion_graphic"):
            # Preset animation mã code hoặc motion graphic.
            # Nếu rơi xuống đây là MathViz fallback:
            if "objects" in cfg:
                from .mathviz.custom_scene import build_custom_scene  # lazy

                sc = build_custom_scene(
                    cfg,
                    make_safe_fn,
                    title=title,
                    subtitle=subtitle,
                    duration=duration,
                )
            else:
                # Tự động dựng 1 procedural scene sống động thay vì trả về None
                sc = build_procedural_fallback_scene(scene, duration)
        else:
            log.info("Preset %r không có cấu hình chi tiết, dùng Procedural Fallback Scene.", preset)
            sc = build_procedural_fallback_scene(scene, duration)
    except Exception as e:  # noqa: BLE001
        log.warning("Dựng animation lỗi (%s): %s, kích hoạt safe procedural fallback", preset, e)
        try:
            sc = build_procedural_fallback_scene(scene, duration)
        except Exception as e2:  # noqa: BLE001
            log.error("Safe procedural fallback cũng lỗi: %s", e2)
            return None

    if sc is None:
        try:
            sc = build_procedural_fallback_scene(scene, duration)
        except Exception:
            return None

    sc.fit_duration(duration)
    return sc


def build_procedural_fallback_scene(scene: ModelScene, duration: float):
    """Tự động dựng 1 MathViz animation scene sống động phù hợp ngữ cảnh của scene.

    ĐẢM BẢO KHÔNG BAO GIỜ TRẢ VỀ NONE: giúp 100% scene yêu cầu animation đều có
    video chuyển động, loại bỏ hoàn toàn việc bị rớt về ảnh tĩnh.
    """
    from .mathviz import scenes as mv_scenes

    title = scene.heading or "Phân Tích Cơ Chế Hoạt Động"
    subtitle = (scene.visual_prompt or scene.narration[:80]).replace("\n", " ").strip()
    text_corpus = f"{scene.heading or ''} {scene.narration or ''} {scene.visual_prompt or ''}".lower()

    # 1. Phát hiện Code / Lệnh / Terminal
    code_indicators = [
        "def ", "class ", "return ", "import ", "select ", "curl ", "docker ",
        "npm ", "function", "const ", "mã lệnh", "thuật toán", "cú pháp"
    ]
    if scene.code_language or any(ci in text_corpus for ci in code_indicators):
        lines = []
        if scene.bullets:
            lines = [b for b in scene.bullets if len(b) < 60][:6]
        if not lines:
            lines = [
                "# Khởi tạo tiến trình xử lý",
                "pipeline = ExecutionPipeline()",
                "result = pipeline.process(stream_data)",
                "if result.status == SUCCESS:",
                "    commit_transaction(result.id)",
                "    notify_cluster_nodes()",
            ]
        return mv_scenes.terminal_scene(
            title=title,
            code_lines=lines,
            language=scene.code_language or "python",
            subtitle=subtitle[:60],
        )

    # 2. Phát hiện So sánh (Comparison / Trade-off)
    comparison_indicators = [
        "so sánh", "vs", "versus", "khác biệt", "trade-off", "đánh đổi",
        "ưu điểm", "nhược điểm", "truyền thống", "tối ưu"
    ]
    if any(ci in text_corpus for ci in comparison_indicators):
        return mv_scenes.comparison_scene(
            title=title,
            left_title="Cách Truyền Thống",
            left_items=["Độ trễ cao", "Nghẽn cổ chai", "Khó mở rộng"],
            right_title="Kiến Trúc Tối Ưu",
            right_items=["Xử lý song song", "Đồng bộ phi tập trung", "Tối ưu bộ nhớ"],
            subtitle=subtitle[:60],
        )

    # 3. Phát hiện Kiến trúc hệ thống / Network / Luồng dữ liệu (Architecture Flow)
    arch_indicators = [
        "kiến trúc", "microservice", "hệ thống", "gateway", "service", "client",
        "server", "database", "redis", "kafka", "luồng", "pipeline", "cluster",
        "node", "packet", "mạng", "phân tán"
    ]
    if any(ai in text_corpus for ai in arch_indicators):
        nodes = []
        if scene.bullets and len(scene.bullets) >= 3:
            nodes = [b[:20] for b in scene.bullets[:5]]
        else:
            nodes = ["Client App", "API Gateway", "Worker Service", "Distributed DB"]
        return mv_scenes.architecture_flow_scene(
            title=title,
            nodes=nodes,
            subtitle=subtitle[:60],
        )

    # 4. Phát hiện Con số / Hiệu năng / Tốc độ (Counter Scene)
    counter_indicators = [
        "triệu", "tỷ", "nghìn", "%", "qps", "tốc độ", "tỉ lệ", "tăng trưởng",
        "gấp", "ms", "latency"
    ]
    if any(ci in text_corpus for ci in counter_indicators):
        import re
        nums = re.findall(r"\b\d+[\.,]?\d*\b", text_corpus)
        val = 100000.0
        unit = "requests/s"
        if nums:
            try:
                val = float(nums[0].replace(",", "."))
                unit = "thông lượng"
            except Exception:
                pass
        return mv_scenes.counter_scene(
            title=title,
            to_value=val,
            unit=unit,
            subtitle=subtitle[:60],
        )

    # 5. Phát hiện Quy trình / Các bước (Steps Scene)
    if scene.bullets and len(scene.bullets) >= 2:
        return mv_scenes.steps_scene(
            title=title,
            steps=[b[:40] for b in scene.bullets[:4]],
            subtitle=subtitle[:60],
        )

    # 6. Mặc định an toàn: Architecture Flow sống động
    default_nodes = ["Yêu Cầu (Input)", "Phân Tích Core", "Xử Lý Logic", "Phản Hồi (Output)"]
    return mv_scenes.architecture_flow_scene(
        title=title,
        nodes=default_nodes,
        subtitle=subtitle[:60],
    )
