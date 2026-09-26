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

    def get_outline(self, n: int = 64) -> list[tuple[float, float]]:
        """Đường bao (scene px): n điểm rải đều quanh chu vi, phục vụ morph."""
        cx, cy = self.bounds_center()
        x0, y0, x1, y1 = self.box
        hw = (x1 - x0) / 2 * self.scale
        hh = (y1 - y0) / 2 * self.scale
        corners = [
            (cx - hw, cy - hh),
            (cx + hw, cy - hh),
            (cx + hw, cy + hh),
            (cx - hw, cy + hh),
        ]
        per = max(1, n // 4)
        pts: list[tuple[float, float]] = []
        for i in range(4):
            ax, ay = corners[i]
            bx, by = corners[(i + 1) % 4]
            for k in range(per):
                f = k / per
                pts.append((ax + (bx - ax) * f + self.dx, ay + (by - ay) * f + self.dy))
        return pts


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

    def get_outline(self, n: int = 64) -> list[tuple[float, float]]:
        """Đường bao (scene px): n điểm rải đều quanh vòng tròn, phục vụ morph."""
        cx = self.center[0] + self.dx
        cy = self.center[1] + self.dy
        r = self.radius * self.scale
        pts: list[tuple[float, float]] = []
        for k in range(n):
            ang = 2 * math.pi * k / n
            pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
        return pts


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
        glow: float = 0.0,
    ) -> None:
        super().__init__()
        self.axes = axes
        self.fn = fn
        self.color = color or THEME.accent
        self.width = width
        self.samples = samples
        self.glow = glow  # 0..1 hào quang neon quanh nét
        self.reveal: float = 1.0  # 0..1 vẽ dần từ trái sang

    def _snapshot_extra(self) -> None:
        self._base_reveal = self.reveal

    def _reset_extra(self) -> None:
        self.reveal = getattr(self, "_base_reveal", 1.0)

    def bounds_center(self) -> tuple[float, float]:
        return self.axes.bounds_center()

    def point_at(self, alpha: float) -> Optional[tuple[float, float]]:
        """Điểm trên đường cong tại tham số alpha∈[0,1] (scene px). None nếu lỗi."""
        xr0, xr1 = self.axes.x_range
        x = xr0 + (xr1 - xr0) * max(0.0, min(1.0, alpha))
        try:
            y = self.fn(x)
        except Exception:  # noqa: BLE001
            return None
        if not math.isfinite(y):
            return None
        px, py = self.axes.to_px(x, y)
        return (px, py)

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
            canvas.polyline(
                pts, self.color, width=self.width, alpha=self._a(), glow=self.glow
            )


# ---------------------------------------------------------------- ParametricCurve
class ParametricCurve(Drawable):
    """Đường cong tham số (x(t), y(t)) trên 1 Axes; vẽ dần qua ``reveal``."""

    def __init__(
        self,
        axes: "Axes",
        fx: Callable[[float], float],
        fy: Callable[[float], float],
        t_range: tuple[float, float] = (0.0, 6.283185307179586),
        color: Optional[str] = None,
        width: float = 5,
        samples: int = 320,
        glow: float = 0.0,
    ) -> None:
        super().__init__()
        self.axes = axes
        self.fx = fx
        self.fy = fy
        self.t_range = t_range
        self.color = color or THEME.accent
        self.width = width
        self.samples = samples
        self.glow = glow
        self.reveal: float = 1.0

    def _snapshot_extra(self) -> None:
        self._base_reveal = self.reveal

    def _reset_extra(self) -> None:
        self.reveal = getattr(self, "_base_reveal", 1.0)

    def bounds_center(self) -> tuple[float, float]:
        return self.axes.bounds_center()

    def point_at(self, alpha: float) -> Optional[tuple[float, float]]:
        """Điểm trên đường cong tham số tại alpha∈[0,1] (scene px). None nếu lỗi."""
        t0, t1 = self.t_range
        t = t0 + (t1 - t0) * max(0.0, min(1.0, alpha))
        try:
            x = self.fx(t)
            y = self.fy(t)
        except Exception:  # noqa: BLE001
            return None
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        px, py = self.axes.to_px(x, y)
        return (px, py)

    def draw(self, canvas: Canvas) -> None:
        t0, t1 = self.t_range
        n = max(2, int(self.samples * max(0.0, min(1.0, self.reveal))))
        pts: list[tuple[float, float]] = []
        for i in range(n):
            t = t0 + (t1 - t0) * i / (self.samples - 1)
            try:
                x = self.fx(t)
                y = self.fy(t)
            except Exception:  # noqa: BLE001
                continue
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            px, py = self.axes.to_px(x, y)
            pts.append((px + self.dx, py + self.dy))
        if len(pts) >= 2:
            canvas.polyline(
                pts, self.color, width=self.width, alpha=self._a(), glow=self.glow
            )


# ---------------------------------------------------------------- Polygon
class Polygon(Drawable):
    """Đa giác đóng từ danh sách đỉnh (scene px). Hỗ trợ morph qua get_outline.

    ``progress`` (0..1) cho phép vẽ dần đường bao như DrawLine; khi <1 chỉ nối
    một phần chu vi (không tô fill). ``_live_pts`` nếu được set (bởi MorphShape)
    sẽ ghi đè hình học tĩnh cho frame đó.
    """

    def __init__(
        self,
        points: Sequence[tuple[float, float]],
        fill: Optional[str] = None,
        outline: Optional[str] = None,
        width: float = 4,
        glow: float = 0.0,
    ) -> None:
        super().__init__()
        self.points = [(float(x), float(y)) for x, y in points]
        self.fill = fill
        self.outline = outline or THEME.accent
        self.width = width
        self.glow = glow
        self.progress: float = 1.0
        self._live_pts: Optional[list[tuple[float, float]]] = None

    def _snapshot_extra(self) -> None:
        self._base_progress = self.progress

    def _reset_extra(self) -> None:
        self.progress = getattr(self, "_base_progress", 1.0)
        self._live_pts = None

    def bounds_center(self) -> tuple[float, float]:
        pts = self.points
        return (
            sum(p[0] for p in pts) / len(pts),
            sum(p[1] for p in pts) / len(pts),
        )

    def get_outline(self, n: int = 64) -> list[tuple[float, float]]:
        """Rải đều n điểm dọc chu vi đa giác (theo chiều dài cạnh), cho morph."""
        src = self.points
        m = len(src)
        seg = [
            math.hypot(src[(i + 1) % m][0] - src[i][0], src[(i + 1) % m][1] - src[i][1])
            for i in range(m)
        ]
        total = sum(seg) or 1.0
        out: list[tuple[float, float]] = []
        for k in range(n):
            d = total * k / n
            acc = 0.0
            for i in range(m):
                if acc + seg[i] >= d or i == m - 1:
                    f = (d - acc) / (seg[i] or 1.0)
                    ax, ay = src[i]
                    bx, by = src[(i + 1) % m]
                    out.append(
                        (ax + (bx - ax) * f + self.dx, ay + (by - ay) * f + self.dy)
                    )
                    break
                acc += seg[i]
        return out

    def draw(self, canvas: Canvas) -> None:
        pts = self._live_pts if self._live_pts is not None else [
            (x + self.dx, y + self.dy) for x, y in self.points
        ]
        if len(pts) < 3:
            return
        p = max(0.0, min(1.0, self.progress))
        if p < 0.999:
            # vẽ dần đường bao (không tô), nối vòng theo tỉ lệ p
            loop = pts + [pts[0]]
            n_edges = len(loop) - 1
            shown = max(2, int(round(n_edges * p)) + 1)
            canvas.polyline(
                loop[:shown], self.outline, width=self.width,
                alpha=self._a(), glow=self.glow,
            )
            return
        canvas.polygon(
            pts,
            fill=self.fill,
            outline=self.outline,
            width=self.width,
            alpha=self._a(),
            glow=self.glow,
        )


# ---------------------------------------------------------------- Formula
class Formula(Drawable):
    """Công thức toán "kiểu LaTeX" render bằng matplotlib mathtext -> ảnh RGBA.

    KHÔNG cần cài LaTeX (mathtext là engine thuần Python của matplotlib). Ảnh
    được cache theo (text,size,color) và ghép lên canvas ở đúng vị trí + anchor.
    Hỗ trợ hiệu ứng ``reveal`` (fade như Write) và scale/opacity như Drawable.
    """

    _CACHE: dict = {}

    def __init__(
        self,
        text: str,
        pos: tuple[float, float],
        size: int = 60,
        color: Optional[str] = None,
        anchor: str = "mm",
    ) -> None:
        super().__init__()
        self.text = text
        self.pos = pos
        self.size = size
        self.color = color or THEME.text
        self.anchor = anchor
        self.reveal: float = 1.0

    def _snapshot_extra(self) -> None:
        self._base_reveal = self.reveal

    def _reset_extra(self) -> None:
        self.reveal = getattr(self, "_base_reveal", 1.0)

    def bounds_center(self) -> tuple[float, float]:
        return self.pos

    @classmethod
    def _render_image(cls, text: str, size: int, color: str, ss: int):
        key = (text, size, color, ss)
        img = cls._CACHE.get(key)
        if img is not None:
            return img
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.mathtext import MathTextParser  # noqa: F401
        except Exception:  # noqa: BLE001
            cls._CACHE[key] = None
            return None
        # bọc trong $...$ nếu người dùng chưa bọc
        s = text.strip()
        if not (s.startswith("$") and s.endswith("$")):
            s = f"${s}$"
        dpi = 200
        fontsize = max(6, int(size * ss * 72 / dpi))
        fig = plt.figure(figsize=(0.01, 0.01), dpi=dpi)
        fig.patch.set_alpha(0.0)
        try:
            t = fig.text(0, 0, s, fontsize=fontsize, color=color)
            fig.canvas.draw()
            bbox = t.get_window_extent(fig.canvas.get_renderer())
            w = max(1, int(math.ceil(bbox.width)) + 8)
            h = max(1, int(math.ceil(bbox.height)) + 8)
            fig.set_size_inches(w / dpi, h / dpi)
            t.set_position((4 / w, 4 / h))
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            rgba = Image.frombuffer(
                "RGBA", fig.canvas.get_width_height(), bytes(buf), "raw", "RGBA", 0, 1
            ).copy()
        except Exception:  # noqa: BLE001
            plt.close(fig)
            cls._CACHE[key] = None
            return None
        plt.close(fig)
        cls._CACHE[key] = rgba
        return rgba

    def draw(self, canvas: Canvas) -> None:
        ss = canvas.ss
        img = self._render_image(self.text, int(self.size * self.scale), self.color, ss)
        if img is None:
            # fallback: vẽ dạng text thường (bỏ ký hiệu $)
            canvas.text(
                (self.pos[0] + self.dx, self.pos[1] + self.dy),
                self.text.strip("$"),
                THEME.font_bold,
                int(self.size * self.scale),
                self.color,
                alpha=self._a(),
                anchor=self.anchor,
            )
            return
        iw, ih = img.size
        if self.reveal < 1.0:
            cut = max(1, int(iw * max(0.0, min(1.0, self.reveal))))
            img = img.crop((0, 0, cut, ih))
            iw = cut
        a = self._a()
        if a < 255:
            alpha = img.split()[3].point(lambda p: int(p * a / 255))
            img = img.copy()
            img.putalpha(alpha)
        # scale ảnh theo zoom camera để công thức cũng zoom/pan cùng cảnh
        zoom = getattr(canvas, "cam_zoom", 1.0)
        if abs(zoom - 1.0) > 1e-3:
            nw = max(1, int(iw * zoom))
            nh = max(1, int(ih * zoom))
            img = img.resize((nw, nh), Image.LANCZOS)
            iw, ih = nw, nh
        # anchor -> góc trái-trên (theo hệ pixel supersample, đã áp camera)
        cx, cy = canvas._px(self.pos[0] + self.dx, self.pos[1] + self.dy)
        h_a = self.anchor[0] if len(self.anchor) >= 1 else "m"
        v_a = self.anchor[1] if len(self.anchor) >= 2 else "m"
        if h_a == "m":
            left = int(cx - iw / 2)
        elif h_a == "r":
            left = int(cx - iw)
        else:
            left = int(cx)
        if v_a == "m":
            top = int(cy - ih / 2)
        elif v_a in ("d", "b", "s"):
            top = int(cy - ih)
        else:
            top = int(cy)
        canvas.paste_rgba(img, (left, top))


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
