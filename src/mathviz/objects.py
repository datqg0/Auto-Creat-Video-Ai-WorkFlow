"""Các Drawable cụ thể: Text, Dot, Line, Arrow, Rect, Circle, Axes,
FunctionGraph, NeuralNet, BarChart, Group.

Mỗi lớp giữ hình học tĩnh + tận dụng thuộc tính động (opacity/scale/dx/dy)
từ Drawable để animation chỉnh. Một số lớp có thêm state động riêng
(vd FunctionGraph.reveal, BarChart.progress) khai báo qua snapshot/reset extra.
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

from PIL import Image, ImageDraw

from .core import Canvas, Drawable
from .theme import THEME, hex_to_rgba, mix


# ---------------------------------------------------------------- Text
class Text(Drawable):
    def __init__(
        self,
        text: str,
        pos: tuple[float, float],
        size: int = 48,
        color: Optional[str] = None,
        bold: bool = True,
        anchor: str = "mm",
        align: str = "center",
    ) -> None:
        super().__init__()
        self.text = text
        self.pos = pos
        self.size = size
        self.color = color or THEME.text
        self.font_path = THEME.font_bold if bold else THEME.font_regular
        self.anchor = anchor
        self.align = align
        # cho hiệu ứng Write: tỉ lệ ký tự hiển thị
        self.reveal: float = 1.0

    def _snapshot_extra(self) -> None:
        self._base_reveal = self.reveal

    def _reset_extra(self) -> None:
        self.reveal = getattr(self, "_base_reveal", 1.0)

    def bounds_center(self) -> tuple[float, float]:
        return self.pos

    def draw(self, canvas: Canvas) -> None:
        shown = self.text
        if self.reveal < 1.0:
            n = max(0, int(round(len(self.text) * self.reveal)))
            shown = self.text[:n]
        if not shown:
            return
        cx, cy = self.pos
        cx += self.dx
        cy += self.dy
        size = int(self.size * self.scale)
        canvas.text(
            (cx, cy),
            shown,
            self.font_path,
            size,
            self.color,
            alpha=self._a(),
            anchor=self.anchor,
            align=self.align,
        )


# ---------------------------------------------------------------- Dot
class Dot(Drawable):
    def __init__(
        self, pos: tuple[float, float], radius: float = 10, color: Optional[str] = None
    ) -> None:
        super().__init__()
        self.pos = pos
        self.radius = radius
        self.color = color or THEME.accent

    def bounds_center(self) -> tuple[float, float]:
        return self.pos

    def draw(self, canvas: Canvas) -> None:
        cx, cy = self.pos[0] + self.dx, self.pos[1] + self.dy
        canvas.circle(
            (cx, cy), self.radius * self.scale, fill=self.color, alpha=self._a()
        )


# ---------------------------------------------------------------- Line
class Line(Drawable):
    def __init__(
        self,
        p1: tuple[float, float],
        p2: tuple[float, float],
        color: Optional[str] = None,
        width: float = 4,
    ) -> None:
        super().__init__()
        self.p1 = p1
        self.p2 = p2
        self.color = color or THEME.text
        self.width = width
        # cho DrawLine: tỉ lệ vẽ dần từ p1 -> p2
        self.progress: float = 1.0

    def _snapshot_extra(self) -> None:
        self._base_progress = self.progress

    def _reset_extra(self) -> None:
        self.progress = getattr(self, "_base_progress", 1.0)

    def bounds_center(self) -> tuple[float, float]:
        return ((self.p1[0] + self.p2[0]) / 2, (self.p1[1] + self.p2[1]) / 2)

    def _end_point(self) -> tuple[float, float]:
        p = max(0.0, min(1.0, self.progress))
        return (
            self.p1[0] + (self.p2[0] - self.p1[0]) * p,
            self.p1[1] + (self.p2[1] - self.p1[1]) * p,
        )

    def draw(self, canvas: Canvas) -> None:
        if self.progress <= 0.001:
            return
        p1 = (self.p1[0] + self.dx, self.p1[1] + self.dy)
        end = self._end_point()
        p2 = (end[0] + self.dx, end[1] + self.dy)
        canvas.line(p1, p2, self.color, width=self.width, alpha=self._a())


# ---------------------------------------------------------------- Arrow
class Arrow(Line):
    def __init__(self, *args, head: float = 22, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.head = head

    def draw(self, canvas: Canvas) -> None:
        if self.progress <= 0.001:
            return
        p1 = (self.p1[0] + self.dx, self.p1[1] + self.dy)
        end = self._end_point()
        p2 = (end[0] + self.dx, end[1] + self.dy)
        canvas.line(p1, p2, self.color, width=self.width, alpha=self._a())
        # đầu mũi tên
        ang = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        h = self.head * self.scale
        for off in (math.radians(150), -math.radians(150)):
            hx = p2[0] + h * math.cos(ang + off)
            hy = p2[1] + h * math.sin(ang + off)
            canvas.line(p2, (hx, hy), self.color, width=self.width, alpha=self._a())


# ---------------------------------------------------------------- Rect
class Rect(Drawable):
    def __init__(
        self,
        box: tuple[float, float, float, float],
        fill: Optional[str] = None,
        outline: Optional[str] = None,
        width: float = 3,
        radius: float = 14,
    ) -> None:
        super().__init__()
        self.box = box
        self.fill = fill
        self.outline = outline
        self.width = width
        self.radius = radius

    def bounds_center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.box
        return ((x0 + x1) / 2, (y0 + y1) / 2)

    def draw(self, canvas: Canvas) -> None:
        cx, cy = self.bounds_center()
        x0, y0, x1, y1 = self.box
        # scale quanh tâm
        hw = (x1 - x0) / 2 * self.scale
        hh = (y1 - y0) / 2 * self.scale
        box = (cx - hw + self.dx, cy - hh + self.dy, cx + hw + self.dx, cy + hh + self.dy)
        canvas.rect(
            box,
            fill=self.fill,
            outline=self.outline,
            width=self.width,
            radius=self.radius,
            alpha=self._a(),
        )


# ---------------------------------------------------------------- Circle
class Circle(Drawable):
    def __init__(
        self,
        center: tuple[float, float],
        radius: float,
        fill: Optional[str] = None,
        outline: Optional[str] = None,
        width: float = 3,
    ) -> None:
        super().__init__()
        self.center = center
        self.radius = radius
        self.fill = fill
        self.outline = outline
        self.width = width

    def bounds_center(self) -> tuple[float, float]:
        return self.center

    def draw(self, canvas: Canvas) -> None:
        c = (self.center[0] + self.dx, self.center[1] + self.dy)
        canvas.circle(
            c,
            self.radius * self.scale,
            fill=self.fill,
            outline=self.outline,
            width=self.width,
            alpha=self._a(),
        )


# ---------------------------------------------------------------- Axes
class Axes(Drawable):
    """Hệ trục toạ độ với vùng vẽ pixel [x0,y0,x1,y1] và miền dữ liệu."""

    def __init__(
        self,
        box: tuple[float, float, float, float],
        x_range: tuple[float, float] = (-1, 1),
        y_range: tuple[float, float] = (-1, 1),
        color: Optional[str] = None,
        grid: bool = True,
        n_ticks: int = 5,
    ) -> None:
        super().__init__()
        self.box = box
        self.x_range = x_range
        self.y_range = y_range
        self.color = color or THEME.muted
        self.grid = grid
        self.n_ticks = n_ticks
        self.z = -1

    def to_px(self, x: float, y: float) -> tuple[float, float]:
        x0, y0, x1, y1 = self.box
        xr0, xr1 = self.x_range
        yr0, yr1 = self.y_range
        px = x0 + (x - xr0) / (xr1 - xr0) * (x1 - x0)
        py = y1 - (y - yr0) / (yr1 - yr0) * (y1 - y0)
        return (px, py)

    def bounds_center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.box
        return ((x0 + x1) / 2, (y0 + y1) / 2)

    def draw(self, canvas: Canvas) -> None:
        x0, y0, x1, y1 = self.box
        a = self._a()
        if self.grid:
            for i in range(self.n_ticks + 1):
                gx = x0 + (x1 - x0) * i / self.n_ticks
                gy = y0 + (y1 - y0) * i / self.n_ticks
                canvas.line((gx, y0), (gx, y1), THEME.grid, width=1, alpha=int(a * 0.6))
                canvas.line((x0, gy), (x1, gy), THEME.grid, width=1, alpha=int(a * 0.6))
        # trục 0 nếu nằm trong miền
        if self.x_range[0] < 0 < self.x_range[1]:
            zx, _ = self.to_px(0, self.y_range[0])
            canvas.line((zx, y0), (zx, y1), self.color, width=2, alpha=a)
        if self.y_range[0] < 0 < self.y_range[1]:
            _, zy = self.to_px(self.x_range[0], 0)
            canvas.line((x0, zy), (x1, zy), self.color, width=2, alpha=a)
        # viền
        canvas.rect((x0, y0, x1, y1), outline=self.color, width=2, radius=0, alpha=a)


# ---------------------------------------------------------------- FunctionGraph
class FunctionGraph(Drawable):
    """Vẽ y=f(x) trên 1 Axes; hỗ trợ vẽ dần bằng thuộc tính reveal."""

    def __init__(
        self,
        axes: Axes,
        fn: Callable[[float], float],
        color: Optional[str] = None,
        width: float = 5,
        samples: int = 240,
    ) -> None:
        super().__init__()
        self.axes = axes
        self.fn = fn
        self.color = color or THEME.accent
        self.width = width
        self.samples = samples
        self.reveal: float = 1.0  # 0..1 vẽ dần từ trái sang

    def _snapshot_extra(self) -> None:
        self._base_reveal = self.reveal

    def _reset_extra(self) -> None:
        self.reveal = getattr(self, "_base_reveal", 1.0)

    def bounds_center(self) -> tuple[float, float]:
        return self.axes.bounds_center()

    def draw(self, canvas: Canvas) -> None:
        xr0, xr1 = self.axes.x_range
        n = max(2, int(self.samples * max(0.0, min(1.0, self.reveal))))
        pts: list[tuple[float, float]] = []
        for i in range(n):
            x = xr0 + (xr1 - xr0) * i / (self.samples - 1)
            try:
                y = self.fn(x)
            except Exception:  # noqa: BLE001
                continue
            if not math.isfinite(y):
                continue
            px, py = self.axes.to_px(x, y)
            pts.append((px + self.dx, py + self.dy))
        if len(pts) >= 2:
            canvas.polyline(pts, self.color, width=self.width, alpha=self._a())


# ---------------------------------------------------------------- NeuralNet
class NeuralNet(Drawable):
    """Mạng neural feed-forward: vẽ node theo lớp + cạnh nối.

    ``pulse`` (0..1) mô phỏng "tín hiệu" chạy từ trái sang phải: cạnh/node
    sáng dần theo vị trí lớp.
    """

    def __init__(
        self,
        layers: Sequence[int],
        box: tuple[float, float, float, float],
        node_radius: float = 26,
        color: Optional[str] = None,
        edge_color: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.layers = list(layers)
        self.box = box
        self.node_radius = node_radius
        self.color = color or THEME.accent
        self.edge_color = edge_color or THEME.grid
        self.pulse: float = 1.0  # 0..1

    def _snapshot_extra(self) -> None:
        self._base_pulse = self.pulse

    def _reset_extra(self) -> None:
        self.pulse = getattr(self, "_base_pulse", 1.0)

    def bounds_center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.box
        return ((x0 + x1) / 2, (y0 + y1) / 2)

    def _positions(self) -> list[list[tuple[float, float]]]:
        x0, y0, x1, y1 = self.box
        cols = len(self.layers)
        out: list[list[tuple[float, float]]] = []
        for ci, count in enumerate(self.layers):
            cx = x0 + (x1 - x0) * (ci / max(1, cols - 1)) if cols > 1 else (x0 + x1) / 2
            col: list[tuple[float, float]] = []
            for ni in range(count):
                if count > 1:
                    cy = y0 + (y1 - y0) * (ni / (count - 1))
                else:
                    cy = (y0 + y1) / 2
                col.append((cx + self.dx, cy + self.dy))
            out.append(col)
        return out

    def draw(self, canvas: Canvas) -> None:
        pos = self._positions()
        a = self._a()
        cols = len(pos)
        # cạnh
        for ci in range(cols - 1):
            layer_frac = (ci + 1) / max(1, cols - 1)
            active = self.pulse >= layer_frac - 0.15
            ec = self.color if active else self.edge_color
            ea = int(a * (0.9 if active else 0.35))
            for p in pos[ci]:
                for q in pos[ci + 1]:
                    canvas.line(p, q, ec, width=2, alpha=ea)
        # node
        for ci, col in enumerate(pos):
            layer_frac = ci / max(1, cols - 1)
            lit = self.pulse >= layer_frac - 0.05
            fill = self.color if lit else THEME.panel
            for p in col:
                canvas.circle(
                    p,
                    self.node_radius * self.scale,
                    fill=fill,
                    outline=self.color,
                    width=3,
                    alpha=a,
                )


# ---------------------------------------------------------------- BarChart
class BarChart(Drawable):
    """Biểu đồ cột động: ``progress`` (0..1) cho cột mọc lên dần."""

    def __init__(
        self,
        values: Sequence[float],
        box: tuple[float, float, float, float],
        labels: Optional[Sequence[str]] = None,
        colors: Optional[Sequence[str]] = None,
        gap_ratio: float = 0.35,
        label_size: int = 30,
    ) -> None:
        super().__init__()
        self.values = list(values)
        self.box = box
        self.labels = list(labels) if labels else None
        self.colors = list(colors) if colors else None
        self.gap_ratio = gap_ratio
        self.label_size = label_size
        self.progress: float = 1.0

    def _snapshot_extra(self) -> None:
        self._base_progress = self.progress

    def _reset_extra(self) -> None:
        self.progress = getattr(self, "_base_progress", 1.0)

    def bounds_center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.box
        return ((x0 + x1) / 2, (y0 + y1) / 2)

    def draw(self, canvas: Canvas) -> None:
        x0, y0, x1, y1 = self.box
        n = len(self.values)
        if n == 0:
            return
        vmax = max(self.values) or 1.0
        total_w = x1 - x0
        slot = total_w / n
        bar_w = slot * (1 - self.gap_ratio)
        a = self._a()
        p = max(0.0, min(1.0, self.progress))
        for i, v in enumerate(self.values):
            color = (self.colors[i] if self.colors else THEME.color(i))
            h = (v / vmax) * (y1 - y0) * p
            bx0 = x0 + slot * i + (slot - bar_w) / 2 + self.dx
            bx1 = bx0 + bar_w
            by1 = y1 + self.dy
            by0 = by1 - h
            canvas.rect((bx0, by0, bx1, by1), fill=color, radius=10, alpha=a)
            if self.labels and i < len(self.labels):
                canvas.text(
                    ((bx0 + bx1) / 2, y1 + 20 + self.dy),
                    self.labels[i],
                    THEME.font_regular,
                    self.label_size,
                    THEME.muted,
                    alpha=a,
                    anchor="ma",
                )


# ---------------------------------------------------------------- Group
class Group(Drawable):
    """Gom nhiều Drawable, animation áp cho cả nhóm (opacity/dx/dy/scale)."""

    def __init__(self, *children: Drawable) -> None:
        super().__init__()
        self.children = list(children)

    def snapshot_base(self) -> None:
        super().snapshot_base()
        for c in self.children:
            c.snapshot_base()

    def reset_state(self) -> None:
        super().reset_state()
        for c in self.children:
            c.reset_state()

    def bounds_center(self) -> tuple[float, float]:
        if not self.children:
            return super().bounds_center()
        cs = [c.bounds_center() for c in self.children]
        return (sum(p[0] for p in cs) / len(cs), sum(p[1] for p in cs) / len(cs))

    def draw(self, canvas: Canvas) -> None:
        for c in self.children:
            # truyền thuộc tính nhóm xuống con
            c.opacity = min(c.opacity, self.opacity)
            c.dx += self.dx
            c.dy += self.dy
            c.draw(canvas)
