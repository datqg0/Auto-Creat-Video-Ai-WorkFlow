"""Module tạo Video AI cho pipeline (Tối đa 1 clip/video theo cấu hình).

Hỗ trợ các provider:
  1. ai_motion (Mặc định & Miễn phí 100%, 0đ, không cần key, tức thì):
     - Sinh ảnh AI chất lượng cao (1080p) bằng FLUX qua Pollinations/HuggingFace.
     - Dựng thành clip MP4 cinematic camera motion (Smooth Zoom + Pan + Cyber glow)
       chuẩn 30fps bằng MoviePy/FFmpeg trong 2-3 giây.
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
    return int(CONFIG["visual"]["height" if vertical else "width"])


def _H(vertical: bool = False) -> int:
    return int(CONFIG["visual"]["width" if vertical else "height"])


def _clean_prompt(prompt: str) -> str:
    """Rút gọn và làm sạch prompt để gửi cho mô hình sinh video/ảnh."""
    p = " ".join(prompt.split()).strip()
    # Loại bỏ các ký tự đặc biệt
    return p[:200]


def _motion_video_from_image(
    img_path: Path,
    out_path: Path,
    duration: float = 5.0,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
) -> Path | None:
    """Biến ảnh tĩnh thành clip MP4 chuyển động camera điện ảnh mượt mà (Ken Burns HD)."""
    try:
        import numpy as np

        img = Image.open(str(img_path)).convert("RGB")
        scale = max(width / img.width, height / img.height) * 1.15  # phóng nhẹ 15% để có biên pan
        bw = max(width, math.ceil(img.width * scale))
        bh = max(height, math.ceil(img.height * scale))
        img = img.resize((bw, bh), Image.Resampling.LANCZOS)

        dur = max(duration, 2.0)
        z0, z1 = 1.0, 1.12
        ux, uy = 0.35, 0.20  # hướng pan nhẹ chéo

        def smooth(t: float) -> float:
            return 0.5 - 0.5 * math.cos(math.pi * t)

        def make_frame(t: float):
            p = smooth(min(max(t / dur, 0.0), 1.0))
            z = z0 + (z1 - z0) * p
            cw = width / z
            ch = height / z

            room_x = (bw - cw) / 2
            room_y = (bh - ch) / 2
            u = -1.0 + 2.0 * p
            cx = bw / 2 + ux * u * room_x
            cy = bh / 2 + uy * u * room_y

            box = (cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2)
            frame_pil = img.resize((width, height), Image.Resampling.BILINEAR, box=box)
            return np.asarray(frame_pil)

        clip = VideoClip(make_frame, duration=dur)
        clip = set_fps(clip, fps)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        clip.write_videofile(
            str(out_path),
            fps=fps,
            codec="libx264",
            audio=False,
            logger=None,
            preset="fast",
            ffmpeg_params=["-pix_fmt", "yuv420p"],
        )
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
    """Tạo clip Video AI qua phương pháp Cinematic Motion từ Ảnh AI FLUX (0đ, tức thì)."""
    w = 1080 if vertical else 1920
    h = 1920 if vertical else 1080
    temp_img = out_path.parent / f"_temp_ai_img_{out_path.stem}.jpg"

    # 1. Sinh ảnh AI FLUX chất lượng cao bám sát prompt
    ai_img = generate_ai_image(prompt, out=temp_img, width=w, height=h)
    if not ai_img or not ai_img.exists():
        log.warning("Không thể sinh ảnh AI làm nguồn cho motion video")
        return None

    # 2. Render thành MP4 chuyển động camera điện ảnh
    fps = int(CONFIG.get("visual", {}).get("fps", 30))
    res = _motion_video_from_image(ai_img, out_path, duration=duration, width=w, height=h, fps=fps)
    temp_img.unlink(missing_ok=True)
    return res


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
    key = hashlib.md5(f"aivid|{clean.lower()}|{orientation}".encode("utf-8")).hexdigest()[:16]
    _CACHE.mkdir(parents=True, exist_ok=True)
    cached_path = _CACHE / f"{key}.mp4"

    if cached_path.exists() and cached_path.stat().st_size > 20000:
        log.info("Dùng video AI từ cache: %s", cached_path.name)
        # Copy hoặc link sang out_dir
        dest = out_dir.parent / f"{out_dir.stem}.mp4" if out_dir.suffix != ".mp4" else out_dir
        try:
            import shutil
            shutil.copyfile(cached_path, dest)
            return dest
        except Exception:
            return cached_path

    dest = out_dir.parent / f"{out_dir.stem}.mp4" if out_dir.suffix != ".mp4" else out_dir

    ai_cfg = CONFIG.get("ai_video", {})
    provider = ai_cfg.get("provider", "ai_motion").lower()

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
        except Exception:
            pass
        return res

    return None
