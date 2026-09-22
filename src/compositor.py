"""Ghép ảnh scene + audio -> clip, nối lại thành video, thêm nhạc nền + phụ đề.

Dùng moviepy (bọc ffmpeg). Mỗi scene = ảnh tĩnh + hiệu ứng zoom nhẹ (Ken Burns)
kéo dài đúng bằng thời lượng audio của scene đó.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path

# Pillow >=10 bỏ Image.ANTIALIAS nhưng moviepy 1.x vẫn gọi -> thêm shim
from PIL import Image as _PILImage

if not hasattr(_PILImage, "ANTIALIAS"):
    _PILImage.ANTIALIAS = _PILImage.Resampling.LANCZOS

from moviepy.editor import (
    AudioFileClip,
    CompositeAudioClip,
    ImageClip,
    concatenate_videoclips,
)

from .config import CONFIG

log = logging.getLogger(__name__)

W = CONFIG["visual"]["width"]
H = CONFIG["visual"]["height"]
FPS = CONFIG["visual"]["fps"]


def _ken_burns(clip: ImageClip, duration: float) -> ImageClip:
    """Zoom nhẹ từ 1.0 -> 1.06 để ảnh tĩnh đỡ nhàm."""
    return clip.resize(lambda t: 1.0 + 0.06 * (t / max(duration, 0.1)))


def _scene_clip(image_path: Path, audio_path: Path) -> ImageClip:
    audio = AudioFileClip(str(audio_path))
    duration = audio.duration
    img = ImageClip(str(image_path)).set_duration(duration)
    img = _ken_burns(img, duration).set_position("center")
    # crop về đúng khung sau khi zoom
    img = img.resize(height=H) if img.h < H else img
    return img.set_audio(audio).set_fps(FPS)


def _pick_music() -> Path | None:
    music_cfg = CONFIG["music"]
    if not music_cfg.get("enabled"):
        return None
    root = Path(__file__).resolve().parent.parent
    mdir = root / music_cfg["directory"]
    if not mdir.exists():
        return None
    tracks = list(mdir.glob("*.mp3")) + list(mdir.glob("*.wav"))
    return random.choice(tracks) if tracks else None


def compose(
    scene_images: list[Path],
    scene_audios: list[Path],
    out_path: Path,
    srt_path: Path | None = None,
) -> Path:
    assert len(scene_images) == len(scene_audios), "Số ảnh và audio phải khớp"

    clips = [_scene_clip(img, aud) for img, aud in zip(scene_images, scene_audios)]
    video = concatenate_videoclips(clips, method="compose")

    # Nhạc nền
    music_path = _pick_music()
    if music_path:
        vol = float(CONFIG["music"].get("volume", 0.12))
        bg = AudioFileClip(str(music_path)).volumex(vol)
        if bg.duration < video.duration:
            from moviepy.audio.fx.all import audio_loop

            bg = audio_loop(bg, duration=video.duration)
        else:
            bg = bg.subclip(0, video.duration)
        video = video.set_audio(CompositeAudioClip([video.audio, bg]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    video.write_videofile(
        str(out_path),
        codec="libx264",
        audio_codec="aac",
        fps=FPS,
        threads=4,
        preset="medium",
        logger=None,
    )
    for c in clips:
        c.close()
    video.close()

    # Burn phụ đề vào video bằng ffmpeg (nếu bật)
    if srt_path and CONFIG["subtitles"].get("burn_in") and srt_path.exists():
        out_path = _burn_subtitles(out_path, srt_path)
    return out_path


def _burn_subtitles(video_path: Path, srt_path: Path) -> Path:
    import subprocess

    burned = video_path.with_name(video_path.stem + "_sub.mp4")
    # escape đường dẫn srt cho filter subtitles
    srt_arg = str(srt_path).replace("\\", "/").replace(":", "\\:")
    style = "FontSize=22,PrimaryColour=&H00FFFFFF,OutlineColour=&H80000000,BorderStyle=3"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"subtitles='{srt_arg}':force_style='{style}'",
            "-c:a",
            "copy",
            str(burned),
        ],
        check=True,
        capture_output=True,
    )
    return burned
