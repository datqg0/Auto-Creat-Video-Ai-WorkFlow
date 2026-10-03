"""Chạy code Python do LLM sinh ra để vẽ animation, rồi trả về file mp4.

An toàn theo cách 2 (sandbox môi trường, KHÔNG blocklist từ khóa):
  - Chạy trong TIẾN TRÌNH CON riêng, ``env`` KHÔNG chứa bất kỳ secret nào
    (chỉ PATH + vài biến vẽ không nhạy cảm). Dù code có đọc os.environ hay
    gọi requests, cũng không có API key/token để rò rỉ.
  - ``timeout`` cứng: code treo/lặp vô hạn bị kill -> không gây hang/OOM cho CI.
  - ``cwd`` là thư mục tạm riêng của scene; dọn theo scene.

Hợp đồng với code AI: runner CHÈN sẵn (trusted) các biến WIDTH/HEIGHT/FPS/
DURATION/OUT_PATH và cấu hình matplotlib Agg + ffmpeg. Code AI chỉ cần vẽ và
lưu animation vào OUT_PATH.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

# Prelude do RUNNER viết (tin cậy) — đặt trước code AI để chuẩn hóa môi trường vẽ.
_PRELUDE = '''\
import matplotlib
matplotlib.use("Agg")
try:
    import imageio_ffmpeg, matplotlib as _mpl
    _mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass
WIDTH = {width}
HEIGHT = {height}
FPS = {fps}
DURATION = {duration}
OUT_PATH = "out.mp4"
# ================= CODE AI BÊN DƯỚI =================
'''


def _safe_env(out_dir: Path) -> dict:
    """Dựng env TỐI THIỂU cho tiến trình con: KHÔNG chứa bất kỳ secret nào.

    PATH để tìm ffmpeg/latex; PYTHONPATH = sys.path để import package đã cài;
    MPLCONFIGDIR cô lập cache matplotlib. Windows cần thêm vài biến hệ thống.
    """
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": os.pathsep.join(p for p in sys.path if p),
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(out_dir),
    }
    if sys.platform == "win32":
        for k in ("SYSTEMROOT", "SYSTEMDRIVE", "TEMP", "TMP", "PATHEXT",
                  "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "APPDATA", "LOCALAPPDATA"):
            if k in os.environ:
                env[k] = os.environ[k]
    else:
        env["HOME"] = str(out_dir)
    return env


def repair_code_with_ai(
    code: str,
    error_msg: str,
    framework: str = "matplotlib",
    duration: float = 5.0,
) -> str | None:
    """Gửi code bị lỗi và traceback cho LLM để tự động phân tích và sửa lỗi."""
    try:
        from .llm import generate_text
    except Exception as e:
        log.warning("Không thể import llm để sửa code: %s", e)
        return None

    prompt = (
        f"You are an expert Python graphics and animation developer specializing in {framework}.\n"
        f"The following script failed to execute during rendering.\n\n"
        f"--- CURRENT SCRIPT ---\n{code}\n\n"
        f"--- ERROR TRACEBACK (STDERR) ---\n{error_msg}\n\n"
        f"TASK:\n"
        f"Fix the code completely so it runs without error.\n"
        f"Target animation duration: {duration} seconds.\n\n"
        f"RULES:\n"
        f"1. Do NOT redefine WIDTH, HEIGHT, FPS, DURATION, OUT_PATH (these are already injected in the runner prelude).\n"
        f"2. Ensure the output animation is saved directly to OUT_PATH ('out.mp4').\n"
        f"3. Only use {framework}, numpy, math, and standard Python libraries.\n"
        f"4. Output ONLY the fixed Python code inside a ```python ... ``` block. No explanations."
    )
    try:
        log.info("Đang gọi AI để tự động sửa lỗi code %s...", framework)
        resp = generate_text(prompt, max_tokens=2048)
        if "```" in resp:
            parts = resp.split("```")
            for i in range(1, len(parts), 2):
                block = parts[i]
                if block.startswith("python"):
                    block = block[6:]
                cleaned = block.strip()
                if cleaned:
                    return cleaned
        return resp.strip() if resp.strip() else None
    except Exception as e:
        log.warning("AI self-repair thất bại: %s", e)
        return None


def generate_matplotlib_from_prompt(
    description: str,
    duration: float = 5.0,
) -> str | None:
    """Yêu cầu AI viết code Matplotlib animation từ mô tả cảnh."""
    try:
        from .llm import generate_text
    except Exception as e:
        log.warning("Không thể import llm: %s", e)
        return None

    prompt = (
        f"Write a high quality Python matplotlib animation script to visualize the following concept:\n"
        f"Concept: {description}\n"
        f"Target duration: {duration} seconds.\n\n"
        f"CRITICAL RULES:\n"
        f"- The runner already injects: WIDTH, HEIGHT, FPS, DURATION, OUT_PATH = 'out.mp4'. DO NOT REDEFINE THEM.\n"
        f"- Use `import matplotlib.pyplot as plt` and `from matplotlib.animation import FuncAnimation`.\n"
        f"- Figure setup: `fig = plt.figure(figsize=(WIDTH/100, HEIGHT/100), dpi=100, facecolor='#0d1117')`.\n"
        f"- Modern dark theme: background '#0d1117', text '#e6edf3', accent '#58a6ff', secondary '#3fb950'.\n"
        f"- Hide axis ticks if not needed for cleaner look: `ax.set_facecolor('#0d1117')`.\n"
        f"- Number of frames = int(DURATION * FPS).\n"
        f"- MUST save animation to OUT_PATH using `ani.save(OUT_PATH, fps=FPS, writer='ffmpeg')`.\n"
        f"- Return ONLY Python code inside ```python ... ``` block."
    )
    try:
        log.info("Đang gọi AI sinh code Matplotlib animation mới...")
        resp = generate_text(prompt, max_tokens=2048)
        if "```" in resp:
            parts = resp.split("```")
            for i in range(1, len(parts), 2):
                block = parts[i]
                if block.startswith("python"):
                    block = block[6:]
                cleaned = block.strip()
                if cleaned:
                    return cleaned
        return resp.strip() if resp.strip() else None
    except Exception as e:
        log.warning("AI sinh code matplotlib thất bại: %s", e)
        return None


def run_ai_code(
    code: str,
    out_dir: Path,
    duration: float,
    width: int,
    height: int,
    fps: int,
    timeout: int = 90,
    auto_repair: bool = True,
    max_repairs: int = 2,
) -> Path | None:
    """Chạy ``code`` (matplotlib) trong sandbox env-rỗng. Tự động AI repair nếu lỗi. Trả về mp4 hoặc None."""
    if not code or not code.strip():
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    script = out_dir / "ai_scene.py"
    safe_env = _safe_env(out_dir)

    current_code = code
    for attempt in range(max_repairs + 1):
        prelude = _PRELUDE.format(
            width=int(width), height=int(height), fps=int(fps), duration=float(duration)
        )
        script.write_text(prelude + current_code, encoding="utf-8")

        try:
            proc = subprocess.run(
                [sys.executable, script.name],
                cwd=str(out_dir),
                env=safe_env,
                timeout=timeout,
                capture_output=True,
                text=True,
            )
        except subprocess.TimeoutExpired:
            log.warning("Code AI vượt timeout %ds (lần %d).", timeout, attempt + 1)
            proc = None
            stderr = f"Execution timed out after {timeout} seconds."
        except Exception as e:  # noqa: BLE001
            log.warning("Chạy code AI lỗi: %s", e)
            proc = None
            stderr = str(e)

        if proc and proc.returncode == 0:
            out = out_dir / "out.mp4"
            if out.exists() and out.stat().st_size > 1024:
                if attempt > 0:
                    log.info("AI Self-Repair Matplotlib thành công ở lần thử %d!", attempt)
                return out
            log.warning("Code AI chạy xong nhưng không sinh out.mp4 hợp lệ.")
            stderr = "No valid out.mp4 produced."
        else:
            stderr = proc.stderr if proc else stderr
            log.warning("Code AI thoát lỗi (lần %d): %s", attempt + 1, (stderr or "")[-500:])

        # Nếu lỗi và còn lượt sửa: gọi AI self-repair
        if auto_repair and attempt < max_repairs:
            log.info("Thử tự động sửa code Matplotlib bằng AI (lượt %d/%d)...", attempt + 1, max_repairs)
            fixed = repair_code_with_ai(current_code, (stderr or "")[-800:], framework="matplotlib", duration=duration)
            if fixed and fixed.strip():
                current_code = fixed
                continue
        break

    return None


# --------------------------------------------------------------------------- manim
# Prelude cho manim: cấu hình độ phân giải/fps/thời lượng nền qua config,
# rồi CLI `manim render` sẽ tìm class Scene đầu tiên trong file.
_MANIM_PRELUDE = '''\
from manim import *
config.frame_rate = {fps}
config.pixel_width = {width}
config.pixel_height = {height}
config.background_color = "{bg}"
# Thời lượng mục tiêu (giây) để code AI canh nhịp animation.
DURATION = {duration}
# ================= CODE AI (MANIM) BÊN DƯỚI =================
'''


def run_manim_code(
    code: str,
    out_dir: Path,
    duration: float,
    width: int,
    height: int,
    fps: int,
    quality: str = "medium_quality",
    background_color: str = "#0d1117",
    timeout: int = 240,
    auto_repair: bool = True,
    max_repairs: int = 2,
) -> Path | None:
    """Chạy ``code`` (manim) trong sandbox env-rỗng. Tự động AI repair nếu lỗi. Trả về mp4 hoặc None."""
    if not code or not code.strip():
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    script = out_dir / "ai_manim.py"
    safe_env = _safe_env(out_dir)
    media = out_dir / "media"

    current_code = code
    for attempt in range(max_repairs + 1):
        prelude = _MANIM_PRELUDE.format(
            width=int(width), height=int(height), fps=int(fps),
            duration=float(duration), bg=background_color,
        )
        script.write_text(prelude + current_code, encoding="utf-8")

        cmd = [
            sys.executable, "-m", "manim", "render",
            "-a", "--format", "mp4",
            f"--quality={_MANIM_QUALITY_FLAG.get(quality, 'm')}",
            "--media_dir", str(media),
            script.name,
        ]
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(out_dir),
                env=safe_env,
                timeout=timeout,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            log.warning("Manim chưa cài -> chuyển hướng fallback.")
            return None
        except subprocess.TimeoutExpired:
            log.warning("Manim vượt timeout %ds (lần %d).", timeout, attempt + 1)
            proc = None
            stderr = f"Execution timed out after {timeout} seconds."
        except Exception as e:  # noqa: BLE001
            log.warning("Chạy manim lỗi: %s", e)
            proc = None
            stderr = str(e)

        if proc and proc.returncode == 0:
            vids = [p for p in media.rglob("*.mp4") if p.stat().st_size > 1024]
            if vids:
                if attempt > 0:
                    log.info("AI Self-Repair Manim thành công ở lần thử %d!", attempt)
                return max(vids, key=lambda p: p.stat().st_size)
            log.warning("Manim chạy xong nhưng không tìm thấy mp4 hợp lệ.")
            stderr = "No valid mp4 generated in media folder."
        else:
            stderr = proc.stderr if proc else stderr
            log.warning("Manim thoát lỗi (lần %d): %s", attempt + 1, (stderr or "")[-800:])

        # Nếu lỗi và còn lượt sửa: gọi AI self-repair
        if auto_repair and attempt < max_repairs:
            log.info("Thử tự động sửa code Manim bằng AI (lượt %d/%d)...", attempt + 1, max_repairs)
            fixed = repair_code_with_ai(current_code, (stderr or "")[-800:], framework="manim", duration=duration)
            if fixed and fixed.strip():
                current_code = fixed
                continue
        break

    return None


_MANIM_QUALITY_FLAG = {
    "low_quality": "l",
    "medium_quality": "m",
    "high_quality": "h",
    "production_quality": "p",
}
