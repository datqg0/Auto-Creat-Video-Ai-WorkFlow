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
