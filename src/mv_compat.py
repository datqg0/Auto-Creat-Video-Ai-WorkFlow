"""Lớp tương thích moviepy 1.x và 2.x.

moviepy 1.x: import từ ``moviepy.editor``, API dùng ``.set_audio``, ``.crossfadein``…
moviepy 2.x: import thẳng từ ``moviepy``, đổi tên thành ``.with_audio``,
             ``.with_duration``, ``CrossFadeIn`` effect…

Module này export tên chung để phần còn lại của dự án không phải quan tâm
đang chạy phiên bản nào. Trên CI (requirements ghim 1.x) và máy local (2.x)
đều chạy được.
"""
from __future__ import annotations

# Pillow >=10 bỏ Image.ANTIALIAS nhưng moviepy 1.x vẫn gọi -> shim
from PIL import Image as _PILImage

if not hasattr(_PILImage, "ANTIALIAS"):
    _PILImage.ANTIALIAS = _PILImage.Resampling.LANCZOS  # type: ignore[attr-defined]

try:  # moviepy 1.x
    from moviepy.editor import (  # type: ignore
        AudioFileClip,
        CompositeAudioClip,
        CompositeVideoClip,
        ImageClip,
        ImageSequenceClip,
        VideoClip,
        VideoFileClip,
        concatenate_videoclips,
    )

    IS_V2 = False
except Exception:  # noqa: BLE001 - moviepy 2.x
    from moviepy import (  # type: ignore
        AudioFileClip,
        CompositeAudioClip,
        CompositeVideoClip,
        ImageClip,
        ImageSequenceClip,
        VideoClip,
        VideoFileClip,
        concatenate_videoclips,
    )

    IS_V2 = True


def set_audio(clip, audio):
    return clip.with_audio(audio) if IS_V2 else clip.set_audio(audio)


def set_duration(clip, dur):
    return clip.with_duration(dur) if IS_V2 else clip.set_duration(dur)


def set_start(clip, t):
    return clip.with_start(t) if IS_V2 else clip.set_start(t)


def set_position(clip, pos):
    return clip.with_position(pos) if IS_V2 else clip.set_position(pos)


def set_fps(clip, fps):
    return clip.with_fps(fps) if IS_V2 else clip.set_fps(fps)


def volumex(clip, factor):
    """Chỉnh âm lượng, tương thích cả 2 bản."""
    if IS_V2:
        from moviepy.audio.fx import MultiplyVolume  # type: ignore

        return clip.with_effects([MultiplyVolume(factor)])
    return clip.volumex(factor)


def crossfadein(clip, dur):
    if IS_V2:
        from moviepy.video.fx import CrossFadeIn  # type: ignore

        return clip.with_effects([CrossFadeIn(dur)])
    return clip.crossfadein(dur)


def crossfadeout(clip, dur):
    if IS_V2:
        from moviepy.video.fx import CrossFadeOut  # type: ignore

        return clip.with_effects([CrossFadeOut(dur)])
    return clip.crossfadeout(dur)


def audio_fadein(clip, dur):
    if IS_V2:
        from moviepy.audio.fx import AudioFadeIn  # type: ignore

        return clip.with_effects([AudioFadeIn(dur)])
    return clip.audio_fadein(dur)


def audio_fadeout(clip, dur):
    if IS_V2:
        from moviepy.audio.fx import AudioFadeOut  # type: ignore

        return clip.with_effects([AudioFadeOut(dur)])
    return clip.audio_fadeout(dur)


def resize(clip, newsize=None, width=None, height=None):
    if IS_V2:
        from moviepy.video.fx import Resize  # type: ignore

        kw = {}
        if newsize is not None:
            kw["new_size"] = newsize
        if width is not None:
            kw["width"] = width
        if height is not None:
            kw["height"] = height
        return clip.with_effects([Resize(**kw)])
    if newsize is not None:
        return clip.resize(newsize)
    return clip.resize(width=width, height=height)


def loop_audio(clip, duration):
    """Lặp audio cho đủ ``duration``."""
    if IS_V2:
        from moviepy.audio.fx import AudioLoop  # type: ignore

        return clip.with_effects([AudioLoop(duration=duration)])
    from moviepy.audio.fx.all import audio_loop  # type: ignore

    return audio_loop(clip, duration=duration)


def loop_video(clip, duration):
    """Lặp video clip cho đủ ``duration`` (b-roll thường ngắn hơn narration)."""
    if IS_V2:
        from moviepy.video.fx import Loop  # type: ignore

        return clip.with_effects([Loop(duration=duration)])
    from moviepy.video.fx.all import loop  # type: ignore

    return loop(clip, duration=duration)


def without_audio(clip):
    """Bỏ audio của clip (b-roll không cần tiếng gốc)."""
    return clip.without_audio() if IS_V2 else clip.set_audio(None)


__all__ = [
    "AudioFileClip",
    "CompositeAudioClip",
    "CompositeVideoClip",
    "ImageClip",
    "ImageSequenceClip",
    "VideoClip",
    "VideoFileClip",
    "concatenate_videoclips",
    "IS_V2",
    "set_audio",
    "set_duration",
    "set_start",
    "set_position",
    "set_fps",
    "volumex",
    "crossfadein",
    "crossfadeout",
    "audio_fadein",
    "audio_fadeout",
    "resize",
    "loop_audio",
    "loop_video",
    "without_audio",
]
