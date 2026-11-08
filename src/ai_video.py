"""Module tạo video từ nhiều ảnh AI hoặc qua dịch vụ video diffusion.

Hỗ trợ các provider:
  1. ai_motion (mặc định):
     - Sinh nhiều ảnh cùng chủ đề qua provider ảnh đã cấu hình.
     - Dựng clip MP4 pan/zoom, chuyển mềm giữa ảnh theo thời lượng lời đọc.
  2. fal (Fal.ai Text-to-Video Diffusion):
     - Gọi Kling, Wan 2.1 hoặc LTX Video qua Fal.ai API nếu có FAL_KEY.
     - Fal.ai tặng credit dùng thử miễn phí khi đăng ký tại https://fal.ai.
  3. pollinations (Pollinations Video):
     - Gọi model Wan/Seedance trên Pollinations nếu có POLLINATIONS_API_KEY có pollen.
  4. Tự động fallback: Nếu provider online lỗi hoặc thiếu key -> tự động fallback
     về ai_motion để đảm bảo pipeline luôn hoàn thành 100%.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
from pathlib import Path
import random
import time
import urllib.parse

from PIL import Image
import requests

from .config import CONFIG, env
from .image_fetcher import generate_ai_image
from .mv_compat import (
    ImageClip,
    VideoClip,
    VideoFileClip,
    set_duration,
    set_fps,
)

log = logging.getLogger(__name__)

_CACHE = Path(__file__).resolve().parent.parent / "assets" / "video_cache"
_TIMEOUT = 120


def _W(vertical: bool = False) -> int:
    return 1080 if vertical else 1920


def _H(vertical: bool = False) -> int:
    return 1920 if vertical else 1080


def _clean_prompt(prompt: str) -> str:
    """Rút gọn và làm sạch prompt để gửi cho mô hình sinh video/ảnh."""
    p = " ".join(prompt.split()).strip()
    # Loại bỏ các ký tự đặc biệt
    return p[:200]


def _motion_video_from_images(
    img_paths: list[Path],
    out_path: Path,
    duration: float = 5.0,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
) -> Path | None:
    """Pan/zoom từng ảnh rồi chuyển mềm giữa các ảnh trong một cảnh."""
    if not img_paths:
        return None
    try:
        import numpy as np

        prepared: list[Image.Image] = []
        for path in img_paths:
            with Image.open(path) as source:
                rgb = source.convert("RGB")
                scale = max(width / rgb.width, height / rgb.height) * 1.15
                bw = max(width, math.ceil(rgb.width * scale))
                bh = max(height, math.ceil(rgb.height * scale))
                prepared.append(rgb.resize((bw, bh), Image.Resampling.LANCZOS))

        dur = max(duration, 2.0)
        segment = dur / len(prepared)
        dissolve = min(0.55, segment * 0.25) if len(prepared) > 1 else 0.0

        def smooth(t: float) -> float:
            return 0.5 - 0.5 * math.cos(math.pi * t)

        def camera_frame(index: int, progress: float) -> Image.Image:
            img = prepared[index]
            p = smooth(min(max(progress, 0.0), 1.0))
            z = 1.0 + 0.10 * p
            cw = width / z
            ch = height / z
            direction = 1 if index % 2 == 0 else -1
            cx = img.width / 2 + direction * 0.32 * (2 * p - 1) * (img.width - cw) / 2
            cy = img.height / 2 + 0.18 * (2 * p - 1) * (img.height - ch) / 2
            box = (cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2)
            return img.resize((width, height), Image.Resampling.BILINEAR, box=box)

        def make_frame(t: float):
            index = min(int(max(t, 0.0) / segment), len(prepared) - 1)
            local = max(t - index * segment, 0.0)
            frame = camera_frame(index, local / segment)
            if index + 1 < len(prepared) and local >= segment - dissolve:
                alpha = min(max((local - segment + dissolve) / dissolve, 0.0), 1.0)
                frame = Image.blend(frame, camera_frame(index + 1, 0.0), alpha)
            return np.asarray(frame)

        clip = VideoClip(make_frame, duration=dur)
        clip = set_fps(clip, fps)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            clip.write_videofile(
                str(out_path), fps=fps, codec="libx264", audio=False,
                logger=None, preset="fast", ffmpeg_params=["-pix_fmt", "yuv420p"],
            )
        finally:
            clip.close()
        if out_path.exists() and out_path.stat().st_size > 10000:
            return out_path
    except Exception as e:
        log.warning("Tạo motion video từ ảnh lỗi: %s", e)
    return None


def _generate_ai_motion_video(
    prompt: str,
    out_path: Path,
    duration: float = 5.0,
    vertical: bool = False,
) -> Path | None:
    """Sinh nhiều ảnh cùng chủ đề, giữ ảnh nguồn và ghép thành một clip."""
    w = 1080 if vertical else 1920
    h = 1920 if vertical else 1080
    ai_cfg = CONFIG.get("ai_video", {}) or {}
    image_count = max(1, min(int(ai_cfg.get("images_per_clip", 3)), 4))
    image_dir = out_path.parent / f"{out_path.stem}_frames"
    image_dir.mkdir(parents=True, exist_ok=True)
    shots = (
        "wide opening view with the main subject clearly visible",
        "closer view of the same subject emphasizing its mechanism",
        "resulting state of the same subject with a clear visual payoff",
        "alternate view of the same subject showing its practical use",
    )
    images: list[Path] = []
    for index in range(image_count):
        image_path = image_dir / f"frame_{index:02d}.jpg"
        shot_prompt = f"{prompt[:160]}. {shots[index]}. Consistent objects and color palette across frames."
        generated = generate_ai_image(shot_prompt, out=image_path, width=w, height=h)
        if generated and generated.exists():
            images.append(generated)
    if not images:
        log.warning("Không thể sinh ảnh AI làm nguồn cho motion video")
        return None

    fps = int(CONFIG.get("visual", {}).get("fps", 30))
    log.info("AI motion: dùng %d/%d ảnh cho %s", len(images), image_count, out_path.name)
    return _motion_video_from_images(images, out_path, duration=duration, width=w, height=h, fps=fps)


def _generate_fal_video(
    prompt: str,
    out_path: Path,
    duration: float = 5.0,
    vertical: bool = False,
) -> Path | None:
    """Sinh video diffusion qua Fal.ai API (Wan 2.1 / Kling / LTX Video).

    Cần FAL_KEY (hoặc FAL_API_KEY) trong .env. Nhận credit dùng thử miễn phí khi đăng ký.
    """
    key = env("FAL_KEY") or env("FAL_API_KEY")
    if not key:
        log.info("Thiếu FAL_KEY -> bỏ qua Fal.ai video")
        return None

    model = CONFIG.get("ai_video", {}).get("fal_model", "fal-ai/wan/v2.1/text-to-video")
    headers = {
        "Authorization": f"Key {key}",
        "Content-Type": "application/json",
    }
    payload = {
        "prompt": prompt,
        "aspect_ratio": "9:16" if vertical else "16:9",
    }

    try:
        log.info("Fal.ai: Đang gửi yêu cầu sinh video model %s...", model)
        submit_url = f"https://queue.fal.run/{model}"
        r = requests.post(submit_url, json=payload, headers=headers, timeout=20)
        r.raise_for_status()
        data = r.json()

        status_url = data.get("status_url")
        response_url = data.get("response_url")
        if not status_url and "video" in data:
            vid_url = data["video"].get("url")
            return _download_video_file(vid_url, out_path)

        # Polling kết quả trong tối đa 120s
        for _ in range(24):
            time.sleep(5)
            chk = requests.get(status_url or response_url, headers=headers, timeout=15)
            if chk.status_code == 200:
                chk_data = chk.json()
                status = chk_data.get("status")
                if status == "COMPLETED" or "video" in chk_data:
                    vid_url = chk_data.get("video", {}).get("url")
                    if not vid_url and response_url:
                        resp = requests.get(response_url, headers=headers, timeout=15)
                        vid_url = resp.json().get("video", {}).get("url")
                    if vid_url:
                        return _download_video_file(vid_url, out_path)
                elif status in ("FAILED", "CANCELLED"):
                    log.warning("Fal.ai render video thất bại: %s", chk_data)
                    return None
    except Exception as e:
        log.warning("Fal.ai video API lỗi: %s", e)
    return None


def _generate_pollinations_video(
    prompt: str,
    out_path: Path,
    duration: float = 5.0,
    vertical: bool = False,
) -> Path | None:
    """Sinh video qua Pollinations API (nếu có POLLINATIONS_API_KEY có pollen credits)."""
    key = env("POLLINATIONS_API_KEY")
    if not key:
        return None

    try:
        model = CONFIG.get("ai_video", {}).get("pollinations_model", "alibaba/wan-2.2-fast")
        url = "https://gen.pollinations.ai/v1/images/generations"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        payload = {
            "prompt": prompt,
            "model": model,
            "size": "1080x1920" if vertical else "1920x1080",
        }
        r = requests.post(url, json=payload, headers=headers, timeout=30)
        if r.status_code == 200:
            res_data = r.json()
            data_arr = res_data.get("data", [])
            if data_arr and "url" in data_arr[0]:
                return _download_video_file(data_arr[0]["url"], out_path)
    except Exception as e:
        log.warning("Pollinations video gen lỗi: %s", e)
    return None


def _download_video_file(url: str, out_path: Path) -> Path | None:
    """Tải file video từ URL về máy."""
    try:
        r = requests.get(url, stream=True, timeout=_TIMEOUT)
        r.raise_for_status()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
        if out_path.exists() and out_path.stat().st_size > 20000:
            return out_path
    except Exception as e:
        log.warning("Tải video file từ %s lỗi: %s", url, e)
    return None


def generate_ai_video(
    prompt: str,
    out_dir: Path,
    duration: float = 5.0,
    vertical: bool = False,
) -> Path | None:
    """Hàm chính: Sinh 1 clip Video AI cho pipeline.

    Kiểm tra cache trước, nếu chưa có thì gọi provider theo cấu hình.
    Tự động fallback về ai_motion nếu provider online gặp lỗi.
    """
    clean = _clean_prompt(prompt)
    if not clean:
        return None

    orientation = "portrait" if vertical else "landscape"
    ai_cfg = CONFIG.get("ai_video", {}) or {}
    provider = ai_cfg.get("provider", "ai_motion").lower()
    image_count = max(1, min(int(ai_cfg.get("images_per_clip", 3)), 4))
    fps = int(CONFIG.get("visual", {}).get("fps", 30))
    cache_spec = f"aivid-v2|{clean.lower()}|{orientation}|{provider}|{duration:.2f}|{image_count}|{fps}"
    key = hashlib.md5(cache_spec.encode("utf-8")).hexdigest()[:16]
    _CACHE.mkdir(parents=True, exist_ok=True)
    cached_path = _CACHE / f"{key}.mp4"

    dest = out_dir.parent / f"{out_dir.stem}.mp4" if out_dir.suffix != ".mp4" else out_dir
    cached_frames = _CACHE / f"{key}_frames"
    dest_frames = dest.parent / f"{dest.stem}_frames"

    if cached_path.exists() and cached_path.stat().st_size > 20000:
        log.info("Dùng video AI từ cache: %s", cached_path.name)
        try:
            import shutil
            shutil.copyfile(cached_path, dest)
            if cached_frames.exists():
                shutil.copytree(cached_frames, dest_frames, dirs_exist_ok=True)
            return dest
        except Exception:
            return cached_path

    res: Path | None = None

    # Thử provider theo ưu tiên
    if provider == "fal":
        res = _generate_fal_video(clean, dest, duration, vertical)
    elif provider == "pollinations":
        res = _generate_pollinations_video(clean, dest, duration, vertical)

    # Fallback hoặc nếu provider mặc định là ai_motion
    if res is None:
        if provider != "ai_motion":
            log.info("Provider '%s' không khả dụng hoặc lỗi -> Fallback về ai_motion", provider)
        res = _generate_ai_motion_video(clean, dest, duration, vertical)

    # Lưu vào cache để tái sử dụng
    if res and res.exists() and res.stat().st_size > 20000:
        try:
            import shutil
            shutil.copyfile(res, cached_path)
            source_frames = res.parent / f"{res.stem}_frames"
            if source_frames.exists():
                shutil.copytree(source_frames, cached_frames, dirs_exist_ok=True)
        except Exception:
            pass
        return res

    return None
