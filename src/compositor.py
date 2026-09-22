"""Ghép ảnh scene + audio -> clip, nối lại thành video, thêm nhạc nền + phụ đề.

Dùng moviepy (bọc ffmpeg). Scene tĩnh = ảnh + hiệu ứng zoom nhẹ (Ken Burns);
scene động = clip mp4 do thư viện mathviz sinh ra. Cả hai đều kéo dài đúng bằng
thời lượng audio của scene đó.

Lớp ``mv_compat`` che khác biệt moviepy 1.x/2.x nên chạy được cả hai bản.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path

from .mv_compat import (
    AudioFileClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    VideoFileClip,
    concatenate_videoclips,
    crossfadein,
    loop_audio,
    loop_video,
    resize,
    set_audio,
    set_duration,
    set_fps,
    set_position,
    set_start,
    volumex,
    without_audio,
)

from .config import CONFIG

log = logging.getLogger(__name__)

W = CONFIG["visual"]["width"]
H = CONFIG["visual"]["height"]
FPS = CONFIG["visual"]["fps"]


def _ken_burns(clip, duration: float):
    """Zoom nhẹ từ 1.0 -> 1.06 để ảnh tĩnh đỡ nhàm."""
    return resize(clip, lambda t: 1.0 + 0.06 * (t / max(duration, 0.1)))


_XFADE = float(CONFIG["visual"].get("crossfade", 0.4))


def _scene_clip(image_path: Path, audio_path: Path):
    """Scene tĩnh: ảnh PNG + Ken Burns + audio."""
    audio = AudioFileClip(str(audio_path))
    duration = audio.duration
    img = set_duration(ImageClip(str(image_path)), duration)
    img = set_position(_ken_burns(img, duration), "center")
    if img.h < H:
        img = resize(img, height=H)
    return set_fps(set_audio(img, audio), FPS)


def _anim_clip(mv_scene, audio_path: Path):
    """Scene động: clip do mathviz sinh + audio narration (đã fit duration)."""
    audio = AudioFileClip(str(audio_path))
    clip = mv_scene.build_clip()
    clip = set_duration(clip, audio.duration)
    return set_fps(set_audio(clip, audio), FPS)


def _cover_video(clip, w: int, h: int):
    """Scale + crop clip phủ kín khung wxh (giữ tỉ lệ, cắt phần thừa)."""
    scale = max(w / clip.w, h / clip.h)
    clip = resize(clip, newsize=(max(1, int(clip.w * scale)), max(1, int(clip.h * scale))))
    x = (clip.w - w) // 2
    y = (clip.h - h) // 2
    if hasattr(clip, "cropped"):
        return clip.cropped(x1=x, y1=y, x2=x + w, y2=y + h)  # moviepy 2.x
    return clip.crop(x1=x, y1=y, x2=x + w, y2=y + h)  # moviepy 1.x


def _broll_clip(video_path: Path, overlay_path: Path | None, audio_path: Path):
    """Scene b-roll: video footage nền (loop cho đủ dài) + overlay chữ + audio narration."""
    audio = AudioFileClip(str(audio_path))
    duration = audio.duration

    bg = without_audio(VideoFileClip(str(video_path)))
    if bg.duration < duration:
        bg = loop_video(bg, duration)
    else:
        bg = bg.subclipped(0, duration) if hasattr(bg, "subclipped") else bg.subclip(0, duration)
    bg = set_duration(_cover_video(bg, W, H), duration)

    layers = [bg]
    if overlay_path and overlay_path.exists():
        ov = set_duration(ImageClip(str(overlay_path), transparent=True), duration)
        layers.append(set_position(ov, (0, 0)))

    comp = CompositeVideoClip(layers, size=(W, H))
    comp = set_duration(comp, duration)
    return set_fps(set_audio(comp, audio), FPS)


def _pick_from(dir_key: str, exts=(".mp3", ".wav")) -> Path | None:
    root = Path(__file__).resolve().parent.parent
    d = root / dir_key
    if not d.exists():
        return None
    files: list[Path] = []
    for e in exts:
        files += list(d.glob(f"*{e}"))
    return random.choice(files) if files else None


def _pick_music() -> Path | None:
    music_cfg = CONFIG["music"]
    if not music_cfg.get("enabled"):
        return None
    return _pick_from(music_cfg["directory"])


def _whoosh() -> Path | None:
    """Hiệu ứng âm thanh chuyển cảnh (nếu có file trong assets/sfx)."""
    sfx_cfg = CONFIG.get("sfx", {})
    if not sfx_cfg.get("enabled"):
        return None
    return _pick_from(sfx_cfg.get("directory", "assets/sfx"))


def compose(
    scene_images: list[Path],
    scene_audios: list[Path],
    out_path: Path,
    srt_path: Path | None = None,
    anim_scenes: list | None = None,
    broll_videos: list | None = None,
    scene_overlays: list | None = None,
) -> Path:
    """Ghép các scene thành video.

    Thứ tự ưu tiên cho mỗi scene:
      1. ``anim_scenes[i]`` (mathviz) nếu khác None -> clip động.
      2. ``broll_videos[i]`` (Path video) nếu khác None -> video footage + overlay chữ.
      3. còn lại -> ảnh tĩnh ``scene_images[i]`` + Ken Burns.
    """
    assert len(scene_images) == len(scene_audios), "Số ảnh và audio phải khớp"
    n = len(scene_images)
    if anim_scenes is None:
        anim_scenes = [None] * n
    if broll_videos is None:
        broll_videos = [None] * n
    if scene_overlays is None:
        scene_overlays = [None] * n

    clips = []
    for img, aud, anim, broll, ov in zip(
        scene_images, scene_audios, anim_scenes, broll_videos, scene_overlays
    ):
        if anim is not None:
            try:
                clips.append(_anim_clip(anim, aud))
                continue
            except Exception as e:  # noqa: BLE001 - fallback về ảnh tĩnh
                log.warning("Render clip động lỗi, dùng ảnh tĩnh: %s", e)
        if broll is not None:
            try:
                clips.append(_broll_clip(broll, ov, aud))
                continue
            except Exception as e:  # noqa: BLE001 - fallback về ảnh tĩnh
                log.warning("Ghép b-roll lỗi, dùng ảnh tĩnh: %s", e)
        clips.append(_scene_clip(img, aud))

    # Chuyển cảnh crossfade nhẹ giữa các scene
    if _XFADE > 0 and len(clips) > 1:
        faded = [clips[0]]
        for c in clips[1:]:
            faded.append(crossfadein(c, _XFADE))
        video = concatenate_videoclips(faded, method="compose", padding=-_XFADE)
    else:
        video = concatenate_videoclips(clips, method="compose")

    audio_layers = [video.audio]

    # Hiệu ứng whoosh tại mỗi điểm chuyển cảnh
    whoosh_path = _whoosh()
    if whoosh_path and len(clips) > 1:
        vol = float(CONFIG.get("sfx", {}).get("volume", 0.3))
        t = 0.0
        for c in clips[:-1]:
            t += c.duration - _XFADE
            try:
                sfx = set_start(volumex(AudioFileClip(str(whoosh_path)), vol), max(t, 0))
                audio_layers.append(sfx)
            except Exception:  # noqa: BLE001
                break

    # Nhạc nền
    music_path = _pick_music()
    if music_path:
        vol = float(CONFIG["music"].get("volume", 0.12))
        bg = volumex(AudioFileClip(str(music_path)), vol)
        if bg.duration < video.duration:
            bg = loop_audio(bg, video.duration)
        else:
            bg = bg.subclipped(0, video.duration) if hasattr(bg, "subclipped") else bg.subclip(0, video.duration)
        audio_layers.append(bg)

    if len(audio_layers) > 1:
        video = set_audio(video, CompositeAudioClip(audio_layers))

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
