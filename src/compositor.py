"""Ghép ảnh scene + audio -> clip, nối lại thành video, thêm nhạc nền + phụ đề.

Dùng moviepy (bọc ffmpeg). Scene tĩnh = ảnh + hiệu ứng zoom nhẹ (Ken Burns);
scene động = clip mp4 do thư viện mathviz sinh ra. Cả hai đều kéo dài đúng bằng
thời lượng audio của scene đó.

Lớp ``mv_compat`` che khác biệt moviepy 1.x/2.x nên chạy được cả hai bản.
"""
from __future__ import annotations

import logging
import math
import random
import zlib
from pathlib import Path

from .mv_compat import (
    AudioFileClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    VideoClip,
    VideoFileClip,
    audio_fadeout,
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
    subclip,
    volumex,
    without_audio,
)

from .config import CONFIG

log = logging.getLogger(__name__)


def _W() -> int:
    return int(CONFIG["visual"]["width"])


def _H() -> int:
    return int(CONFIG["visual"]["height"])


def _FPS() -> int:
    return int(CONFIG["visual"]["fps"])


def _get_watermark_sprite() -> "Image.Image | None":
    """Tạo sprite watermark thương hiệu góc màn hình (nhẹ, tái sử dụng)."""
    from PIL import Image, ImageDraw, ImageFont

    brand_cfg = CONFIG.get("branding", {})
    wm_cfg = brand_cfg.get("watermark", {})
    if wm_cfg.get("enabled", True) is False:
        return None

    text = wm_cfg.get("text", "TECH LAB | AI")
    font_p = Path(__file__).resolve().parent.parent / CONFIG["visual"].get("font", "assets/fonts/BeVietnamPro-Bold.ttf")
    try:
        font = ImageFont.truetype(str(font_p), 13)
    except Exception:
        font = ImageFont.load_default(13)

    # Đo kích thước chữ để đóng khung vừa vặn
    dummy = Image.new("RGBA", (1, 1))
    d_dum = ImageDraw.Draw(dummy)
    tw = int(d_dum.textlength(text, font=font))
    badge_w = tw + 46
    badge_h = 32

    sprite = Image.new("RGBA", (badge_w, badge_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(sprite)
    # Khung kính mờ công nghệ viền xanh cyan
    d.rounded_rectangle([0, 0, badge_w - 1, badge_h - 1], radius=8, fill=(15, 23, 42, 175), outline=(56, 189, 248, 130), width=1)
    # Đèn LED xanh lá neon trạng thái online
    d.ellipse([10, 11, 18, 19], fill=(57, 255, 20, 255))
    d.text((26, 8), text, font=font, fill=(226, 232, 240, 230))
    return sprite


def _get_particle_sprites() -> list["Image.Image"]:
    """Tạo bộ sprite hạt nano cyber phát sáng đa màu."""
    from PIL import Image, ImageDraw

    colors = [
        (0, 240, 255),    # Cyan
        (57, 255, 20),    # Neon green
        (168, 85, 247),   # Purple
        (255, 230, 100),  # Gold
    ]
    sprites = []
    for col in colors:
        dot = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
        d = ImageDraw.Draw(dot)
        d.ellipse([1, 1, 14, 14], fill=(*col, 40))
        d.ellipse([4, 4, 11, 11], fill=(*col, 180))
        d.ellipse([6, 6, 9, 9], fill=(255, 255, 255, 220))
        sprites.append(dot)
    return sprites


def _ken_burns_clip(image_path: Path, duration: float):
    """Ảnh tĩnh -> clip động: zoom + pan có easing và beat snapping, kết hợp
    micro-motion (hạt cyber phát sáng trôi lơ lửng, quét scanline nhẹ holographic,
    watermark nhận diện thương hiệu góc trên).
    """
    import numpy as np
    from PIL import Image

    from .mathviz.easing import smooth

    img = Image.open(str(image_path)).convert("RGB")
    scale = max(_W() / img.width, _H() / img.height)
    if abs(scale - 1.0) > 1e-3:
        img = img.resize(
            (max(_W(), math.ceil(img.width * scale)), max(_H(), math.ceil(img.height * scale))),
            Image.LANCZOS,
        )
    bw, bh = img.size

    rng = random.Random(zlib.crc32(image_path.name.encode("utf-8")))
    z_lo, z_hi = 1.03, 1.08
    z0, z1 = (z_lo, z_hi) if rng.random() < 0.6 else (z_hi, z_lo)
    ang = rng.uniform(0, 2 * math.pi)
    ux, uy = math.cos(ang) * 0.5, math.sin(ang) * 0.5
    dur = max(duration, 0.1)

    # Chuẩn bị micro-motion: 20 hạt cyber nhẹ nhàng bay lên
    dot_sprites = _get_particle_sprites()
    particles = []
    for _ in range(20):
        particles.append({
            "x": rng.uniform(0.04, 0.96),
            "y": rng.uniform(0.04, 0.96),
            "vx": rng.uniform(-0.012, 0.012),
            "vy": rng.uniform(-0.035, -0.010),
            "sprite_idx": rng.randint(0, len(dot_sprites) - 1),
        })

    wm_sprite = _get_watermark_sprite()
    W, H = _W(), _H()

    def frame(t: float):
        x = min(max(t / dur, 0.0), 1.0)
        # Beat & rhythm snapping: nhịp mở đầu scene có punch deceleration nhẹ trong 0.35s
        punch = 0.022 * math.exp(-t * 5.0)
        p = 0.35 * x + 0.65 * smooth(x)
        z = z0 + (z1 - z0) * p + punch
        cw, ch = W / z, H / z
        room_x, room_y = (bw - cw) / 2, (bh - ch) / 2
        u = -1.0 + 2.0 * p
        cx = bw / 2 + ux * u * room_x
        cy = bh / 2 + uy * u * room_y
        box = (cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2)

        # Cắt và zoom ảnh nền
        frame_pil = img.resize((W, H), Image.BILINEAR, box=box).convert("RGBA")

        # 1. Micro-motion: Hạt nano cyber trôi lơ lửng
        for pt in particles:
            px = int(((pt["x"] + pt["vx"] * t) % 1.0) * (W - 20))
            py = int(((pt["y"] + pt["vy"] * t) % 1.0) * (H - 20))
            spr = dot_sprites[pt["sprite_idx"]]
            frame_pil.paste(spr, (px, py), spr)

        # 2. Watermark thương hiệu (góc trên bên phải)
        if wm_sprite is not None:
            wm_x = W - wm_sprite.width - 32
            frame_pil.paste(wm_sprite, (wm_x, 32), wm_sprite)

        # 3. Quét tia scanline holographic dịu mắt (chu kỳ 5 giây)
        arr = np.asarray(frame_pil.convert("RGB")).copy()
        scan_y = int(((t / 5.0) % 1.0) * H)
        for dy in range(-8, 9):
            y = scan_y + dy
            if 0 <= y < H:
                intensity = int(14 * (1.0 - abs(dy) / 9.0))
                arr[y, :, 0] = np.clip(arr[y, :, 0].astype(np.int16) + intensity // 3, 0, 255)
                arr[y, :, 1] = np.clip(arr[y, :, 1].astype(np.int16) + intensity, 0, 255)
                arr[y, :, 2] = np.clip(arr[y, :, 2].astype(np.int16) + intensity, 0, 255)

        return arr

    return VideoClip(frame, duration=duration)


_XFADE = float(CONFIG["visual"].get("crossfade", 0.4))


def _scene_clip(image_path: Path, audio_path: Path):
    """Scene tĩnh: ảnh PNG + Ken Burns (zoom + pan) + audio."""
    audio = AudioFileClip(str(audio_path))
    duration = audio.duration
    img = _ken_burns_clip(image_path, duration)
    return set_fps(set_audio(img, audio), _FPS())


def _anim_clip(mv_scene, audio_path: Path):
    """Scene động: clip do mathviz sinh + audio narration (đã fit duration)."""
    audio = AudioFileClip(str(audio_path))
    clip = mv_scene.build_clip()
    clip = set_duration(clip, audio.duration)
    return set_fps(set_audio(clip, audio), _FPS())


def _cover_video(clip, w: int, h: int):
    """Scale + crop clip phủ kín khung wxh (giữ tỉ lệ, cắt phần thừa)."""
    scale = max(w / clip.w, h / clip.h)
    clip = resize(clip, newsize=(max(1, int(clip.w * scale)), max(1, int(clip.h * scale))))
    x = (clip.w - w) // 2
    y = (clip.h - h) // 2
    if hasattr(clip, "cropped"):
        return clip.cropped(x1=x, y1=y, x2=x + w, y2=y + h)  # moviepy 2.x
    return clip.crop(x1=x, y1=y, x2=x + w, y2=y + h)  # moviepy 1.x


def _code_clip(video_path: Path, audio_path: Path):
    """Scene do CODE AI sinh: mp4 render sẵn + audio narration, phủ kín khung."""
    audio = AudioFileClip(str(audio_path))
    duration = audio.duration
    src = VideoFileClip(str(video_path))  # giữ nguồn để đóng sau, tránh rò rỉ ffmpeg reader
    bg = without_audio(src)
    if bg.duration < duration:
        bg = loop_video(bg, duration)
    else:
        try:
            target_dur = min(duration, bg.duration)
            bg = bg.subclipped(0, target_dur) if hasattr(bg, "subclipped") else bg.subclip(0, target_dur)
        except Exception as e:
            log.warning("subclip code clip thất bại (%s), dùng clip gốc", e)
    bg = set_duration(_cover_video(bg, _W(), _H()), duration)
    out = set_fps(set_audio(bg, audio), _FPS())
    out._src_clips = [src, audio]  # moviepy không đóng đệ quy -> tự dọn ở compose()
    return out


def _broll_clip(video_path: Path, overlay_path: Path | None, audio_path: Path):
    """Scene b-roll: video footage nền (loop cho đủ dài) + overlay chữ + audio narration."""
    audio = AudioFileClip(str(audio_path))
    duration = audio.duration

    src = VideoFileClip(str(video_path))  # giữ nguồn để đóng sau, tránh rò rỉ ffmpeg reader
    bg = without_audio(src)
    if bg.duration < duration:
        bg = loop_video(bg, duration)
    else:
        try:
            target_dur = min(duration, bg.duration)
            bg = bg.subclipped(0, target_dur) if hasattr(bg, "subclipped") else bg.subclip(0, target_dur)
        except Exception as e:
            log.warning("subclip b-roll thất bại (%s), dùng clip gốc", e)
    bg = set_duration(_cover_video(bg, _W(), _H()), duration)

    layers = [bg]
    if overlay_path and overlay_path.exists():
        ov = set_duration(ImageClip(str(overlay_path), transparent=True), duration)
        layers.append(set_position(ov, (0, 0)))

    comp = CompositeVideoClip(layers, size=(_W(), _H()))
    comp = set_duration(comp, duration)
    out = set_fps(set_audio(comp, audio), _FPS())
    out._src_clips = [src, audio]  # moviepy không đóng đệ quy -> tự dọn ở compose()
    return out


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
    code_videos: list | None = None,
) -> Path:
    """Ghép các scene thành video.

    Thứ tự ưu tiên cho mỗi scene:
      1. ``code_videos[i]`` (mp4 do code AI sinh) nếu khác None -> clip code.
      2. ``anim_scenes[i]`` (mathviz) nếu khác None -> clip động.
      3. ``broll_videos[i]`` (Path video) nếu khác None -> video footage + overlay chữ.
      4. còn lại -> ảnh tĩnh ``scene_images[i]`` + Ken Burns.
    """
    assert len(scene_images) == len(scene_audios), "Số ảnh và audio phải khớp"
    n = len(scene_images)
    if anim_scenes is None:
        anim_scenes = [None] * n
    if broll_videos is None:
        broll_videos = [None] * n
    if scene_overlays is None:
        scene_overlays = [None] * n
    if code_videos is None:
        code_videos = [None] * n

    clips = []
    for img, aud, anim, broll, ov, codev in zip(
        scene_images, scene_audios, anim_scenes, broll_videos, scene_overlays, code_videos
    ):
        if codev is not None:
            try:
                clips.append(_code_clip(codev, aud))
                continue
            except Exception as e:  # noqa: BLE001 - fallback về ảnh tĩnh
                log.warning("Ghép clip code AI lỗi, dùng ảnh tĩnh: %s", e)
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

    # Mốc bắt đầu audio của từng scene (tuần tự) - dùng cho cả video lẫn whoosh.
    starts: list[float] = []
    t = 0.0
    for c in clips:
        starts.append(t)
        t += c.duration
    total = t

    if _XFADE > 0 and len(clips) > 1:
        # Audio GIỮ tuần tự (đúng thời lượng) để tránh "giọng đi trước hình".
        # Video crossfade kiểu "hình dẫn": mỗi cảnh hiện đủ NGAY KHI lời bắt đầu
        # (fade hoàn tất tại mốc starts[i], nơi narration scene i mới cất lời).
        seq_audio = [set_start(c.audio, starts[i]) for i, c in enumerate(clips)]
        video_layers = []
        for i, c in enumerate(clips):
            v = without_audio(c)
            try:
                v = set_duration(v, c.duration + _XFADE)  # phần đuôi để chồng dissolve
            except Exception:  # noqa: BLE001 - clip không kéo dài được -> giữ nguyên
                pass
            if i == 0:
                video_layers.append(set_start(v, 0.0))
            else:
                video_layers.append(set_start(crossfadein(v, _XFADE), max(starts[i] - _XFADE, 0.0)))
        video = set_duration(CompositeVideoClip(video_layers, size=(_W(), _H())), total)
        video = set_audio(video, CompositeAudioClip(seq_audio))
    else:
        video = concatenate_videoclips(clips, method="compose")

    audio_layers = [video.audio]

    # Hiệu ứng whoosh tại mỗi ranh giới scene (đồng bộ với lời)
    whoosh_path = _whoosh()
    if whoosh_path and len(clips) > 1:
        vol = float(CONFIG.get("sfx", {}).get("volume", 0.3))
        for st in starts[1:]:
            try:
                sfx = set_start(volumex(AudioFileClip(str(whoosh_path)), vol), max(st - _XFADE, 0))
                audio_layers.append(sfx)
            except Exception:  # noqa: BLE001
                break

    # Nhạc nền: KHÔNG trộn ở đây nữa. Để pass ffmpeg cuối (_finalize) trộn bằng
    # sidechaincompress -> nhạc tự động nhỏ lại khi có giọng (ducking thật).
    music_path = _pick_music() if CONFIG.get("music", {}).get("enabled") else None

    if len(audio_layers) > 1:
        video = set_audio(video, CompositeAudioClip(audio_layers))

    # Bảo vệ trần thời lượng cho Short: YouTube Shorts giới hạn cứng 3 phút (180 giây).
    # Nếu dài > 180s, YouTube sẽ chuyển thành video thông thường và mất tab Shorts.
    cur_dur = getattr(video, "duration", None) or total
    if CONFIG.get("active_mode") == "short" and cur_dur > 180.0:
        log.warning(
            "Thời lượng Short (%.1fs) vượt trần 180s của YouTube Shorts, tự động cắt về 179.0s để đảm bảo định dạng Short.",
            cur_dur,
        )
        video = subclip(video, 0, 179.0)
        if getattr(video, "audio", None):
            try:
                video = set_audio(video, audio_fadeout(video.audio, 0.5))
            except Exception as e:  # noqa: BLE001
                log.warning("Audio fade-out cho Short lỗi, bỏ qua: %s", e)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    video.write_videofile(
        str(out_path),
        codec="libx264",
        audio_codec="aac",
        fps=_FPS(),
        threads=4,
        preset="slow",
        # CRF 18 = gần lossless, hết banding vùng gradient/chữ; yuv420p cho YouTube.
        ffmpeg_params=["-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        logger=None,
    )
    for c in clips:
        for s in getattr(c, "_src_clips", []):
            try:
                s.close()
            except Exception:  # noqa: BLE001
                pass
        c.close()
    video.close()

    # Burn phụ đề + chuẩn hóa âm lượng giọng (loudnorm) + trộn nhạc ducking ở pass cuối.
    burn = bool(srt_path and CONFIG["subtitles"].get("burn_in") and srt_path.exists())
    return _finalize(out_path, srt_path if burn else None, music_path)


def _finalize(video_path: Path, srt_path: Path | None, music_path: Path | None) -> Path:
    """Pass ffmpeg cuối: loudnorm giọng (-14 LUFS) + ducking nhạc nền + burn phụ đề."""
    import subprocess

    out = video_path.with_name(video_path.stem + "_final.mp4")
    music_vol = float(CONFIG.get("music", {}).get("volume", 0.12))

    cmd = ["ffmpeg", "-y", "-i", str(video_path)]
    if music_path is not None:
        cmd += ["-stream_loop", "-1", "-i", str(music_path)]

    # Chuỗi filter audio: chuẩn giọng -> (nếu có nhạc) nhạc bị nén theo giọng rồi amix.
    if music_path is not None:
        af = (
            "[0:a]loudnorm=I=-14:TP=-1.5:LRA=11,asplit=2[voice][vkey];"
            f"[1:a]volume={music_vol}[bg];"
            "[bg][vkey]sidechaincompress=threshold=0.03:ratio=8:attack=5:release=350[duck];"
            "[voice][duck]amix=inputs=2:duration=first:dropout_transition=0,"
            "loudnorm=I=-14:TP=-1.5:LRA=11[aout]"
        )
        audio_map = ["-filter_complex", af, "-map", "0:v", "-map", "[aout]"]
    else:
        audio_map = ["-af", "loudnorm=I=-14:TP=-1.5:LRA=11"]

    if srt_path is not None:
        srt_arg = str(srt_path).replace("\\", "/").replace(":", "\\:")
        # Font Việt đậm, viền đen, nền mờ bán trong suốt, sát đáy (MarginV) dễ đọc mọi nền.
        style = (
            "FontName=Be Vietnam Pro,Fontsize=24,Bold=1,"
            "PrimaryColour=&H00FFFFFF,OutlineColour=&HC8000000,BackColour=&H99000000,"
            "BorderStyle=4,Outline=2,Shadow=1,MarginV=60,Alignment=2"
        )
        # fontsdir để libass nạp font Việt cục bộ (CI không cài sẵn Be Vietnam Pro).
        fonts_dir = (Path(__file__).resolve().parent.parent / "assets" / "fonts")
        fdir_arg = str(fonts_dir).replace("\\", "/").replace(":", "\\:")
        vf = f"subtitles='{srt_arg}':fontsdir='{fdir_arg}':force_style='{style}'"
        video_enc = ["-vf", vf, "-c:v", "libx264", "-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p"]
    else:
        video_enc = ["-c:v", "copy"]

    cmd += audio_map + video_enc + ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except Exception as e:  # noqa: BLE001 - lỗi pass cuối -> trả video gốc
        log.warning("Pass finalize (loudnorm/duck/subtitle) lỗi, dùng video gốc: %s", e)
        return video_path
    return out
