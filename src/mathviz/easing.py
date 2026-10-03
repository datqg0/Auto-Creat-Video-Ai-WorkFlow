"""Hàm easing + nội suy — nền tảng cho mọi animation.

Tất cả hàm easing nhận t trong [0,1] và trả về giá trị [0,1] (thường vậy),
dùng để "làm mượt" chuyển động thay vì tuyến tính khô cứng.
"""
from __future__ import annotations

import math
from typing import Sequence


# ---------- Easing ----------

def linear(t: float) -> float:
    return t


def smooth(t: float) -> float:
    """Smoothstep — chậm ở hai đầu, nhanh ở giữa (giống 'smooth' của Manim)."""
    t = _clamp(t)
    return t * t * (3 - 2 * t)


def smoother(t: float) -> float:
    """Smootherstep — mượt hơn nữa (đạo hàm bậc 2 cũng liên tục)."""
    t = _clamp(t)
    return t * t * t * (t * (t * 6 - 15) + 10)


def ease_in(t: float) -> float:
    t = _clamp(t)
    return t * t


def ease_out(t: float) -> float:
    t = _clamp(t)
    return 1 - (1 - t) * (1 - t)


def ease_in_out(t: float) -> float:
    t = _clamp(t)
    if t < 0.5:
        return 2 * t * t
    return 1 - (-2 * t + 2) ** 2 / 2


def ease_out_back(t: float) -> float:
    """Vọt qua rồi lùi lại chút — cảm giác nảy nhẹ (dùng cho GrowFromCenter)."""
    t = _clamp(t)
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2


def ease_out_elastic(t: float) -> float:
    t = _clamp(t)
    if t == 0 or t == 1:
        return t
    c4 = (2 * math.pi) / 3
    return 2 ** (-10 * t) * math.sin((t * 10 - 0.75) * c4) + 1


def ease_out_bounce(t: float) -> float:
    t = _clamp(t)
    n1 = 7.5625
    d1 = 2.75
    if t < 1 / d1:
        return n1 * t * t
    if t < 2 / d1:
        t -= 1.5 / d1
        return n1 * t * t + 0.75
    if t < 2.5 / d1:
        t -= 2.25 / d1
        return n1 * t * t + 0.9375
    t -= 2.625 / d1
    return n1 * t * t + 0.984375


# ---------- Nội suy ----------

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp_point(
    p1: Sequence[float], p2: Sequence[float], t: float
) -> tuple[float, float]:
    return (lerp(p1[0], p2[0], t), lerp(p1[1], p2[1], t))


def lerp_color(
    c1: Sequence[int], c2: Sequence[int], t: float
) -> tuple[int, ...]:
    return tuple(int(round(lerp(c1[i], c2[i], t))) for i in range(len(c1)))


def ease_out_expo(t: float) -> float:
    """Vọt rất nhanh rồi dừng dứt khoát — tạo cảm giác punchy dứt điểm."""
    t = _clamp(t)
    return 1.0 if t >= 1.0 else 1.0 - (2.0 ** (-10.0 * t))


def ease_in_expo(t: float) -> float:
    t = _clamp(t)
    return 0.0 if t <= 0.0 else 2.0 ** (10.0 * (t - 1.0))


def ease_in_out_expo(t: float) -> float:
    t = _clamp(t)
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    if t < 0.5:
        return (2.0 ** (20.0 * t - 10.0)) / 2.0
    return (2.0 - 2.0 ** (-20.0 * t + 10.0)) / 2.0


def spring_step(t: float, damping: float = 8.0, freq: float = 12.0) -> float:
    """Mô phỏng springStep (dao động tắt dần qua đích rồi snap ổn định)."""
    t = _clamp(t)
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return 1.0 - math.exp(-damping * t) * math.cos(freq * t)


def _clamp(t: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, t))


EASINGS = {
    "linear": linear,
    "smooth": smooth,
    "smoother": smoother,
    "ease_in": ease_in,
    "ease_out": ease_out,
    "ease_in_out": ease_in_out,
    "ease_out_expo": ease_out_expo,
    "ease_in_expo": ease_in_expo,
    "ease_in_out_expo": ease_in_out_expo,
    "expo": ease_out_expo,
    "spring": spring_step,
    "spring_step": spring_step,
    "back": ease_out_back,
    "elastic": ease_out_elastic,
    "bounce": ease_out_bounce,
}


def get_easing(name):
    """Nhận tên (str) hoặc callable, trả về callable easing."""
    if callable(name):
        return name
    return EASINGS.get(name, smooth)
