"""Core của mathviz: Canvas + Scene + timeline animation.

Mô hình:
- Mọi phần tử hình học là 1 ``Drawable`` với các thuộc tính "động"
  (opacity, scale, ...) mà animation sẽ chỉnh theo thời gian.
- ``Scene`` giữ 1 timeline: ``add`` để hiện phần tử tĩnh, ``play`` để chạy
  animation (đẩy playhead tiến lên), ``wait`` để dừng.
- Khi render 1 frame tại thời điểm t, ta *mô phỏng lại* từ đầu: reset trạng
  thái gốc rồi áp mọi animation có start <= t. Nhờ vậy random-access frame
  luôn tất định (deterministic), dễ xuất mp4 song song hay preview 1 khung.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .theme import THEME, hex_to_rgb, hex_to_rgba

log = logging.getLogger(__name__)

_FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    key = (path, size)
    f = _FONT_CACHE.get(key)
    if f is None:
        try:
            f = ImageFont.truetype(path, size)
        except Exception:  # noqa: BLE001
            try:
                f = ImageFont.load_default(size)
            except Exception:  # noqa: BLE001
                f = ImageFont.load_default()
        _FONT_CACHE[key] = f
    return f


class Canvas:
    """Bọc 1 PIL Image RGBA + tiện ích vẽ (anti-alias bằng supersampling)."""

    def __init__(self, width: int, height: int, bg: str, ss: int = 1) -> None:
        self.width = width
        self.height = height
        self.ss = max(1, int(ss))  # hệ số supersample cho nét mượt
        self.bg = bg
        w, h = width * self.ss, height * self.ss
        self.img = Image.new("RGBA", (w, h), hex_to_rgba(bg, 255))
        self.draw = ImageDraw.Draw(self.img, "RGBA")

    # --- toạ độ: nhân theo supersample ---
    def _s(self, v: float) -> float:
        return v * self.ss

    def font(self, path: str, size: int) -> ImageFont.FreeTypeFont:
        return load_font(path, int(size * self.ss))

    # --- primitives ---
    def line(
        self,
        p1: tuple[float, float],
        p2: tuple[float, float],
        color: str,
        width: float = 3,
        alpha: int = 255,
    ) -> None:
        self.draw.line(
            [self._s(p1[0]), self._s(p1[1]), self._s(p2[0]), self._s(p2[1])],
            fill=hex_to_rgba(color, alpha),
            width=max(1, int(self._s(width))),
        )

    def polyline(
        self,
        pts: list[tuple[float, float]],
        color: str,
        width: float = 3,
        alpha: int = 255,
    ) -> None:
        if len(pts) < 2:
            return
        flat = [self._s(v) for pt in pts for v in pt]
        self.draw.line(
            flat, fill=hex_to_rgba(color, alpha), width=max(1, int(self._s(width))), joint="curve"
        )

    def circle(
        self,
        center: tuple[float, float],
        radius: float,
        fill: Optional[str] = None,
        outline: Optional[str] = None,
        width: float = 2,
        alpha: int = 255,
    ) -> None:
        cx, cy, r = self._s(center[0]), self._s(center[1]), self._s(radius)
        box = [cx - r, cy - r, cx + r, cy + r]
        self.draw.ellipse(
            box,
            fill=hex_to_rgba(fill, alpha) if fill else None,
            outline=hex_to_rgba(outline, alpha) if outline else None,
            width=max(1, int(self._s(width))) if outline else 1,
        )

    def rect(
        self,
        box: tuple[float, float, float, float],
        fill: Optional[str] = None,
        outline: Optional[str] = None,
        width: float = 2,
        radius: float = 0,
        alpha: int = 255,
    ) -> None:
        x0, y0, x1, y1 = (self._s(v) for v in box)
        f = hex_to_rgba(fill, alpha) if fill else None
        o = hex_to_rgba(outline, alpha) if outline else None
        w = max(1, int(self._s(width))) if outline else 1
        if radius > 0:
            self.draw.rounded_rectangle(
                [x0, y0, x1, y1], radius=self._s(radius), fill=f, outline=o, width=w
            )
        else:
            self.draw.rectangle([x0, y0, x1, y1], fill=f, outline=o, width=w)

    def text(
        self,
        pos: tuple[float, float],
        text: str,
        font_path: str,
        size: int,
        color: str,
        alpha: int = 255,
        anchor: str = "la",
        align: str = "left",
    ) -> None:
        self.draw.text(
            (self._s(pos[0]), self._s(pos[1])),
            text,
            font=self.font(font_path, size),
            fill=hex_to_rgba(color, alpha),
            anchor=anchor,
            align=align,
        )

    def text_size(
        self, text: str, font_path: str, size: int
    ) -> tuple[float, float]:
        f = self.font(font_path, size)
        box = self.draw.textbbox((0, 0), text, font=f)
        return ((box[2] - box[0]) / self.ss, (box[3] - box[1]) / self.ss)

    def paste_rgba(self, overlay: Image.Image, box: tuple[int, int]) -> None:
        """Ghép 1 lớp RGBA (đã ở kích thước supersample) lên canvas."""
        self.img.alpha_composite(overlay, dest=box)

    def finalize(self) -> Image.Image:
        """Trả về ảnh RGB kích thước gốc (downscale nếu có supersample)."""
        img = self.img
        if self.ss > 1:
            img = img.resize((self.width, self.height), Image.LANCZOS)
        return img.convert("RGB")


class Drawable:
    """Phần tử hình học cơ bản. Con kế thừa và cài ``draw``.

    Thuộc tính động chung: opacity (0..1), scale, dx/dy (dịch chuyển thêm).
    Animation sẽ chỉnh các thuộc tính này. Trước mỗi frame, Scene gọi
    ``reset_state`` để về giá trị gốc rồi mới áp animation.
    """

    def __init__(self) -> None:
        self.opacity: float = 1.0
        self.scale: float = 1.0
        self.dx: float = 0.0
        self.dy: float = 0.0
        # trạng thái gốc để reset mỗi frame
        self._base: dict[str, float] = {}
        self.z: int = 0  # thứ tự vẽ (nhỏ vẽ trước)
        self._visible_from: float = 0.0

    def snapshot_base(self) -> None:
        """Lưu giá trị gốc của các thuộc tính động (gọi 1 lần khi add/play)."""
        self._base = {
            "opacity": self.opacity,
            "scale": self.scale,
            "dx": self.dx,
            "dy": self.dy,
        }
        self._snapshot_extra()

    def _snapshot_extra(self) -> None:  # con override nếu có thêm state động
        pass

    def reset_state(self) -> None:
        for k, v in self._base.items():
            setattr(self, k, v)
        self._reset_extra()

    def _reset_extra(self) -> None:
        pass

    # tiện ích cho con: alpha 0..255 theo opacity hiện tại
    def _a(self, extra: float = 1.0) -> int:
        return max(0, min(255, int(round(self.opacity * extra * 255))))

    def bounds_center(self) -> tuple[float, float]:
        """Tâm hình học — mặc định override ở con để scale/pulse đúng chỗ."""
        return (THEME.width / 2, THEME.height / 2)

    def draw(self, canvas: Canvas) -> None:  # pragma: no cover - abstract
        raise NotImplementedError


class Animation:
    """Base animation: chỉnh thuộc tính target theo alpha (đã qua easing)."""

    def __init__(self, target, run_time: float = 1.0, easing="smooth") -> None:
        from .easing import get_easing

        self.target = target
        self.run_time = max(0.001, float(run_time))
        self.easing = get_easing(easing)
        self.start: float = 0.0

    @property
    def end(self) -> float:
        return self.start + self.run_time

    def alpha_at(self, t: float) -> float:
        raw = (t - self.start) / self.run_time
        raw = max(0.0, min(1.0, raw))
        return self.easing(raw)

    def apply(self, alpha: float) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def apply_at(self, t: float) -> None:
        if t < self.start:
            return
        self.apply(self.alpha_at(t))


class Scene:
    """Timeline động: add / play / wait rồi render_mp4 hoặc render_png."""

    def __init__(
        self,
        duration: Optional[float] = None,
        bg: Optional[str] = None,
        ss: int = 2,
    ) -> None:
        self.width = THEME.width
        self.height = THEME.height
        self.fps = THEME.fps
        self.bg = bg or THEME.bg
        self.ss = ss
        self._objects: list[Drawable] = []
        self._anims: list[Animation] = []
        self.playhead: float = 0.0
        self._fixed_duration = duration
        self._bg_layers: list[Callable[[Canvas], None]] = []

    # ---------- xây timeline ----------
    def add(self, *objs: Drawable) -> "Scene":
        for o in objs:
            o._visible_from = self.playhead
            o.snapshot_base()
            self._objects.append(o)
        return self

    def add_background(self, fn: Callable[[Canvas], None]) -> "Scene":
        """Thêm hàm vẽ nền tuỳ biến (gọi trước khi vẽ objects mỗi frame)."""
        self._bg_layers.append(fn)
        return self

    def play(self, *anims: Animation, run_time: Optional[float] = None) -> "Scene":
        """Chạy 1 hoặc nhiều animation song song, đẩy playhead theo cái dài nhất."""
        longest = 0.0
        for a in anims:
            if run_time is not None:
                a.run_time = run_time
            a.start = self.playhead
            if a.target is not None and a.target not in self._objects:
                a.target._visible_from = self.playhead
                a.target.snapshot_base()
                self._objects.append(a.target)
            self._anims.append(a)
            longest = max(longest, a.run_time)
        self.playhead += longest
        return self

    def wait(self, dt: float = 1.0) -> "Scene":
        self.playhead += max(0.0, dt)
        return self

    @property
    def duration(self) -> float:
        if self._fixed_duration is not None:
            return max(self._fixed_duration, 0.1)
        end = self.playhead
        for a in self._anims:
            end = max(end, a.end)
        return max(end, 0.1)

    def fit_duration(self, target: float) -> "Scene":
        """Kéo/giãn để tổng thời lượng đúng bằng target (khớp audio narration).

        Cách làm: nếu timeline ngắn hơn target thì thêm wait cuối; nếu dài hơn
        thì scale toàn bộ start/run_time cho vừa. Đơn giản, đủ dùng.
        """
        cur = self.duration
        if target <= 0:
            return self
        if cur <= target + 1e-6:
            self._fixed_duration = target
            return self
        factor = target / cur
        for a in self._anims:
            a.start *= factor
            a.run_time *= factor
        self.playhead *= factor
        self._fixed_duration = target
        return self

    # ---------- render ----------
    def _render_frame(self, t: float) -> Image.Image:
        canvas = Canvas(self.width, self.height, self.bg, ss=self.ss)
        for fn in self._bg_layers:
            fn(canvas)
        # reset trạng thái động rồi áp animation tới thời điểm t
        for o in self._objects:
            o.reset_state()
        for a in self._anims:
            a.apply_at(t)
        for o in sorted(self._objects, key=lambda x: x.z):
            if t + 1e-6 < o._visible_from:
                continue
            if o.opacity <= 0.001:
                continue
            o.draw(canvas)
        return canvas.finalize()

    def render_png(self, out_path, t: Optional[float] = None) -> Path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tt = self.duration * 0.6 if t is None else t
        self._render_frame(tt).save(out)
        return out

    def frames(self):
        """Generator numpy RGB frame cho moviepy (dùng khi tích hợp)."""
        n = max(1, int(round(self.duration * self.fps)))
        for i in range(n):
            t = i / self.fps
            yield np.asarray(self._render_frame(t))

    def build_clip(self):
        """Trả về moviepy VideoClip (ImageSequenceClip) — để compositor dùng."""
        from ..mv_compat import ImageSequenceClip  # lazy

        frames = list(self.frames())
        return ImageSequenceClip(frames, fps=self.fps)

    def render_mp4(self, out_path, audio_path: Optional[str] = None) -> Path:
        """Xuất mp4. Cần moviepy. audio_path (wav) sẽ được ghép nếu có."""
        from ..mv_compat import AudioFileClip, set_audio, set_duration  # lazy

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        clip = self.build_clip()
        audio_codec = None
        if audio_path and Path(audio_path).exists():
            aud = AudioFileClip(str(audio_path))
            clip = set_duration(set_audio(clip, aud), min(clip.duration, aud.duration))
            audio_codec = "aac"
        clip.write_videofile(
            str(out),
            codec="libx264",
            audio_codec=audio_codec,
            fps=self.fps,
            preset="medium",
            logger=None,
        )
        return out
