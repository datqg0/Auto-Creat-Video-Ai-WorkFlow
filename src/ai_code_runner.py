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


def run_ai_code(
    code: str,
    out_dir: Path,
    duration: float,
    width: int,
    height: int,
    fps: int,
    timeout: int = 90,
) -> Path | None:
    """Chạy ``code`` (matplotlib) trong sandbox env-rỗng. Trả về mp4 hoặc None.

    Code AI được kỳ vọng lưu animation vào biến ``OUT_PATH`` (= 'out.mp4' trong
    ``out_dir``). None nếu lỗi/timeout/không sinh được file hợp lệ.
    """
    if not code or not code.strip():
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    script = out_dir / "ai_scene.py"
    prelude = _PRELUDE.format(
        width=int(width), height=int(height), fps=int(fps), duration=float(duration)
    )
    script.write_text(prelude + code, encoding="utf-8")

    # env TỐI THIỂU: không truyền secret. PATH để tìm ffmpeg; MPLBACKEND phòng hờ.
    # PYTHONPATH = sys.path để Python con import được package đã cài (không cần APPDATA).
    # MPLCONFIGDIR cô lập cache matplotlib vào thư mục tạm (khỏi cần HOME thật).
    safe_env = _safe_env(out_dir)

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
        log.warning("Code AI vượt timeout %ds -> bỏ.", timeout)
        return None
    except Exception as e:  # noqa: BLE001
        log.warning("Chạy code AI lỗi: %s", e)
        return None

    if proc.returncode != 0:
        log.warning("Code AI thoát mã %s: %s", proc.returncode, (proc.stderr or "")[-500:])
        return None

    out = out_dir / "out.mp4"
    if out.exists() and out.stat().st_size > 1024:
        return out
    log.warning("Code AI chạy xong nhưng không sinh out.mp4 hợp lệ.")
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
) -> Path | None:
    """Chạy ``code`` (manim) trong sandbox env-rỗng. Trả về mp4 hoặc None.

    Code AI được kỳ vọng định nghĩa MỘT class kế thừa ``Scene`` (manim tự tìm và
    render class đầu tiên). Render bằng CLI ``manim`` với ``env`` KHÔNG chứa secret,
    có ``timeout`` cứng. None nếu manim chưa cài / lỗi / timeout / không ra mp4.
    """
    if not code or not code.strip():
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    script = out_dir / "ai_manim.py"
    prelude = _MANIM_PRELUDE.format(
        width=int(width), height=int(height), fps=int(fps),
        duration=float(duration), bg=background_color,
    )
    script.write_text(prelude + code, encoding="utf-8")

    safe_env = _safe_env(out_dir)
    media = out_dir / "media"
    # -a: render mọi Scene trong file; --format mp4 để không ra webm/gif.
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
        log.warning("Manim chưa cài -> bỏ (fallback pycode/custom).")
        return None
    except subprocess.TimeoutExpired:
        log.warning("Manim vượt timeout %ds -> bỏ.", timeout)
        return None
    except Exception as e:  # noqa: BLE001
        log.warning("Chạy manim lỗi: %s", e)
        return None

    if proc.returncode != 0:
        log.warning("Manim thoát mã %s: %s", proc.returncode, (proc.stderr or "")[-800:])
        return None

    # Manim ghi mp4 vào media/videos/<script>/<res>/<SceneName>.mp4. Lấy file
    # mp4 mới nhất & lớn nhất trong media_dir để không phụ thuộc tên class.
    vids = [p for p in media.rglob("*.mp4") if p.stat().st_size > 1024]
    if not vids:
        log.warning("Manim chạy xong nhưng không tìm thấy mp4 hợp lệ.")
        return None
    return max(vids, key=lambda p: p.stat().st_size)


_MANIM_QUALITY_FLAG = {
    "low_quality": "l",
    "medium_quality": "m",
    "high_quality": "h",
    "production_quality": "p",
}
