"""Animation cụ thể: FadeIn/Out, Write, DrawLine, GrowFromCenter, Move,
CountUp, Pulse.

Mỗi animation chỉnh 1 vài thuộc tính động của target dựa trên alpha
(đã qua easing) trong khoảng [0,1].
"""
from __future__ import annotations

import math
from typing import Optional

from .core import Animation, Drawable


class FadeIn(Animation):
    def __init__(self, target: Drawable, run_time: float = 0.8, shift: float = 0.0, easing="smooth"):
        super().__init__(target, run_time, easing)
        self.shift = shift  # dịch lên khi hiện (px)

    def apply(self, alpha: float) -> None:
        self.target.opacity = self.target._base.get("opacity", 1.0) * alpha
        if self.shift:
            self.target.dy = self.target._base.get("dy", 0.0) + self.shift * (1 - alpha)


class FadeOut(Animation):
    def __init__(self, target: Drawable, run_time: float = 0.8, shift: float = 0.0, easing="smooth"):
        super().__init__(target, run_time, easing)
        self.shift = shift

    def apply(self, alpha: float) -> None:
        self.target.opacity = self.target._base.get("opacity", 1.0) * (1 - alpha)
        if self.shift:
            self.target.dy = self.target._base.get("dy", 0.0) - self.shift * alpha


class Write(Animation):
    """Hiện chữ dần theo ký tự (target cần có .reveal, vd Text)."""

    def __init__(self, target, run_time: float = 1.2, easing="linear"):
        super().__init__(target, run_time, easing)

    def apply(self, alpha: float) -> None:
        self.target.opacity = 1.0
        if hasattr(self.target, "reveal"):
            self.target.reveal = alpha


class DrawLine(Animation):
    """Vẽ dần Line/Arrow/FunctionGraph từ đầu tới cuối (target cần .progress)."""

    def __init__(self, target, run_time: float = 1.0, easing="smooth"):
        super().__init__(target, run_time, easing)

    def apply(self, alpha: float) -> None:
        self.target.opacity = 1.0
        if hasattr(self.target, "progress"):
            self.target.progress = alpha
        elif hasattr(self.target, "reveal"):
            self.target.reveal = alpha


class GrowFromCenter(Animation):
    """Phình từ 0 -> kích thước thật, có nảy nhẹ (back easing)."""

    def __init__(self, target: Drawable, run_time: float = 0.7, easing="back"):
        super().__init__(target, run_time, easing)

    def apply(self, alpha: float) -> None:
        base = self.target._base.get("scale", 1.0)
        self.target.scale = base * alpha
        self.target.opacity = self.target._base.get("opacity", 1.0) * min(1.0, alpha * 1.5)


class Move(Animation):
    """Dịch chuyển tương đối (dx, dy) so với vị trí gốc."""

    def __init__(
        self,
        target: Drawable,
        dx: float = 0.0,
        dy: float = 0.0,
        run_time: float = 1.0,
        easing="ease_in_out",
    ):
        super().__init__(target, run_time, easing)
        self.mx = dx
        self.my = dy

    def apply(self, alpha: float) -> None:
        self.target.dx = self.target._base.get("dx", 0.0) + self.mx * alpha
        self.target.dy = self.target._base.get("dy", 0.0) + self.my * alpha


class CountUp(Animation):
    """Đếm số từ from_value -> to_value, ghi vào target.text (target là Text)."""

    def __init__(
        self,
        target,
        from_value: float,
        to_value: float,
        run_time: float = 1.5,
        fmt: str = "{:.0f}",
        easing="smooth",
    ):
        super().__init__(target, run_time, easing)
        self.from_value = from_value
        self.to_value = to_value
        self.fmt = fmt

    def apply(self, alpha: float) -> None:
        self.target.opacity = 1.0
        val = self.from_value + (self.to_value - self.from_value) * alpha
        self.target.text = self.fmt.format(val)


class Pulse(Animation):
    """Nhấp nháy/scale nhẹ theo sin để nhấn mạnh (lặp n lần trong run_time)."""

    def __init__(
        self,
        target: Drawable,
        amount: float = 0.15,
        cycles: float = 1.0,
        run_time: float = 0.8,
        easing="linear",
    ):
        super().__init__(target, run_time, easing)
        self.amount = amount
        self.cycles = cycles

    def apply(self, alpha: float) -> None:
        base = self.target._base.get("scale", 1.0)
        s = math.sin(alpha * math.pi * self.cycles)
        self.target.scale = base * (1 + self.amount * s)


class Signal(Animation):
    """Chạy tín hiệu trong NeuralNet: chỉnh target.pulse 0 -> 1."""

    def __init__(self, target, run_time: float = 1.5, easing="smooth"):
        super().__init__(target, run_time, easing)

    def apply(self, alpha: float) -> None:
        self.target.opacity = 1.0
        if hasattr(self.target, "pulse"):
            self.target.pulse = alpha


class CameraMove(Animation):
    """Zoom/pan camera: nội suy zoom + tâm từ giá trị hiện tại tới đích.

    Không nhắm vào Drawable mà vào ``Scene.camera`` (bind khi play). Đặt
    ``is_camera=True`` để Scene biết gắn camera thay vì thêm object.
    """

    is_camera = True

    def __init__(
        self,
        zoom: float = 1.0,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        run_time: float = 1.2,
        easing="ease_in_out",
    ):
        super().__init__(None, run_time, easing)
        self.to_zoom = zoom
        self.to_cx = cx
        self.to_cy = cy
        self._cam = None
        self._from = None

    def bind(self, camera) -> None:
        self._cam = camera
        self._from = (camera.zoom, camera.cx, camera.cy)
        # cập nhật base camera để bước tiếp theo nối tiếp từ đích
        camera.zoom = self.to_zoom
        camera.cx = self.to_cx if self.to_cx is not None else camera.cx
        camera.cy = self.to_cy if self.to_cy is not None else camera.cy
        camera.snapshot_base()

    def apply(self, alpha: float) -> None:
        if self._cam is None or self._from is None:
            return
        fz, fcx, fcy = self._from
        tz = self.to_zoom
        tcx = self.to_cx if self.to_cx is not None else fcx
        tcy = self.to_cy if self.to_cy is not None else fcy
        self._cam.zoom = fz + (tz - fz) * alpha
        self._cam.cx = fcx + (tcx - fcx) * alpha
        self._cam.cy = fcy + (tcy - fcy) * alpha


class MoveAlongPath(Animation):
    """Di chuyển target (thường là Dot) chạy dọc theo 1 đường cong.

    ``path`` là đối tượng có ``point_at(alpha)`` -> (px, py) scene: dùng
    FunctionGraph hoặc ParametricCurve. Dot sẽ bám theo path khi alpha 0->1.
    Nếu ``trace=True``, đồng thời "vẽ dần" path (chỉnh path.reveal) để nét cong
    mọc ra ngay dưới điểm đang chạy — hiệu ứng "đầu bút" rất đẹp.
    """

    def __init__(self, target, path, run_time: float = 1.5, trace: bool = False, easing="ease_in_out"):
        super().__init__(target, run_time, easing)
        self.path = path
        self.trace = trace
        # đảm bảo path được đăng ký vào scene (để trace vẽ dần hoạt động)
        self.extra_targets = [path]

    def apply(self, alpha: float) -> None:
        self.target.opacity = self.target._base.get("opacity", 1.0)
        pt = None
        if hasattr(self.path, "point_at"):
            pt = self.path.point_at(alpha)
        if pt is not None:
            base_pos = getattr(self.target, "pos", None)
            if base_pos is not None:
                self.target.dx = self.target._base.get("dx", 0.0) + (pt[0] - base_pos[0])
                self.target.dy = self.target._base.get("dy", 0.0) + (pt[1] - base_pos[1])
        if self.trace and hasattr(self.path, "reveal"):
            self.path.reveal = alpha


class Transform(Animation):
    """Morph "hình" nguồn -> đích: nội suy vị trí/kích thước + cross-fade.

    Dùng chung cho Circle/Rect/Dot/Text: khi alpha tăng, ``source`` co/di tới
    dáng của ``dest`` và mờ dần, đồng thời ``dest`` hiện dần. Cách tiếp cận
    "tween + cross-fade" đủ mượt cho hình cơ bản mà không cần khớp path phức tạp.
    """

    def __init__(self, source, dest, run_time: float = 1.0, easing="smooth"):
        super().__init__(source, run_time, easing)
        self.dest = dest
        self.extra_targets = [dest]
        self._sc = source.bounds_center()
        self._dc = dest.bounds_center()

    def apply(self, alpha: float) -> None:
        # source: trôi tới tâm đích + mờ dần
        sc, dc = self._sc, self._dc
        self.target.dx = self.target._base.get("dx", 0.0) + (dc[0] - sc[0]) * alpha
        self.target.dy = self.target._base.get("dy", 0.0) + (dc[1] - sc[1]) * alpha
        self.target.opacity = self.target._base.get("opacity", 1.0) * (1 - alpha)
        # dest: hiện dần tại chỗ
        self.dest.opacity = self.dest._base.get("opacity", 1.0) * alpha


class MorphShape(Animation):
    """Biến hình THỰC (vertex morph) A -> B qua khớp điểm đường bao.

    Khác Transform (chỉ tween tâm + cross-fade), MorphShape lấy get_outline()
    của cả nguồn & đích, rải cùng số điểm, xoay danh sách căn theo tâm để giảm
    "xoắn", rồi nội suy tuyến tính từng đỉnh -> hình trung gian mượt (kiểu Manim).

    Target phải là 1 ``Polygon`` (nơi ghi _live_pts mỗi frame). ``source`` và
    ``dest`` là bất kỳ đối tượng nào có ``get_outline(n)`` (Circle/Rect/Polygon).
    ``dest`` được ẩn trong lúc morph; kết thúc alpha=1 thì trùng khít dest.
    """

    def __init__(self, target, source, dest, run_time: float = 1.2,
                 n: int = 96, easing="smooth"):
        super().__init__(target, run_time, easing)
        self.source = source
        self.dest = dest
        self.n = max(12, int(n))
        self.extra_targets = [dest]
        self._src = self._aligned(source)
        self._dst = self._aligned(dest)

    def _aligned(self, obj) -> list[tuple[float, float]]:
        """Lấy outline n điểm; xoay list để đỉnh 0 gần hướng +x so với tâm."""
        pts = list(obj.get_outline(self.n))
        if len(pts) < 3:
            return pts
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        best = min(
            range(len(pts)),
            key=lambda i: abs(math.atan2(pts[i][1] - cy, pts[i][0] - cx)),
        )
        return pts[best:] + pts[:best]

    def apply(self, alpha: float) -> None:
        src, dst = self._src, self._dst
        m = min(len(src), len(dst))
        if m < 3:
            return
        live = [
            (
                src[i][0] + (dst[i][0] - src[i][0]) * alpha,
                src[i][1] + (dst[i][1] - src[i][1]) * alpha,
            )
            for i in range(m)
        ]
        self.target._live_pts = live
        self.target.opacity = self.target._base.get("opacity", 1.0)
        # ẩn dest trong lúc morph, chỉ hiện đúng lúc gần kết thúc
        self.dest.opacity = self.dest._base.get("opacity", 1.0) * max(
            0.0, (alpha - 0.98) / 0.02
        )
