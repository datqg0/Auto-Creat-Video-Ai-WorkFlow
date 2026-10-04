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

    if framework == "manim":
        framework_rules = (
            "1. Output ONE Scene class `class NeonScene(Scene):`.\n"
            "2. Do NOT redefine neon helpers: neon, NeonDot, neon_trail, sym, neon_box, curve, make_grid, send, hud (already injected).\n"
            "3. NO LaTeX: do NOT use Tex, MathTex, DecimalNumber. Use sym(...) or Text for symbols.\n"
            "4. Keep coordinates inside safe frame: x in [-6.2, 6.2], y in [-3.3, 3.3].\n"
            "5. Output ONLY the fixed Python code inside a ```python ... ``` block. No explanations."
        )
    else:
        framework_rules = (
            "1. Do NOT redefine WIDTH, HEIGHT, FPS, DURATION, OUT_PATH (these are already injected in the runner prelude).\n"
            "2. Ensure the output animation is saved directly to OUT_PATH ('out.mp4').\n"
            "3. Only use matplotlib, numpy, math, and standard Python libraries.\n"
            "4. Output ONLY the fixed Python code inside a ```python ... ``` block. No explanations."
        )

    prompt = (
        f"You are an expert Python graphics and animation developer specializing in {framework}.\n"
        f"The following script failed to execute during rendering.\n\n"
        f"--- CURRENT SCRIPT ---\n{code}\n\n"
        f"--- ERROR TRACEBACK (STDERR) ---\n{error_msg}\n\n"
        f"TASK:\n"
        f"Fix the code completely so it runs without error.\n"
        f"Target animation duration: {duration} seconds.\n\n"
        f"RULES:\n"
        f"{framework_rules}"
    )
    try:
        log.info("Đang gọi AI để tự động sửa lỗi code %s...", framework)
        resp = generate_text(prompt, max_tokens=2048, task="code")
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
        log.info("Đang gọi AI chuyên code sinh Matplotlib animation mới...")
        resp = generate_text(prompt, max_tokens=2048, task="code")
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


def generate_manim_from_prompt(
    description: str,
    duration: float = 5.0,
) -> str | None:
    """Yêu cầu AI viết code Manim animation từ mô tả cảnh / thuật toán visual."""
    try:
        from .llm import generate_text
    except Exception as e:
        log.warning("Không thể import llm: %s", e)
        return None

    prompt = (
        f"You are a creative-coding expert. Build a SINGLE-FILE Manim (Community Edition) animation with a "
        f"beautiful LIGHT NEON look (glowing lines on dark background).\n"
        f"The scene must be PURELY VISUAL: the picture explains itself. NO descriptive sentences or long text.\n\n"
        f"TOPIC / CONCEPT TO VISUALIZE:\n"
        f"{description}\n\n"
        f"TARGET DURATION: {duration:.1f} seconds.\n\n"
        f"ENVIRONMENT & PRELUDE (ALREADY INJECTED - DO NOT REDEFINE OR RE-IMPORT):\n"
        f"- Manim, numpy, DURATION={duration:.1f}, BG='#05060f' are already loaded.\n"
        f"- Palette: CY='#00f0ff' (cyan), PK='#ff2bd6' (pink), GR='#39ff14' (green), "
        f"YE='#ffe600' (yellow), PU='#9d4dff' (purple), OR='#ff8a00' (orange).\n"
        f"- Neon Helpers provided (ready to use):\n"
        f"  * neon(mob, color=CY, width=3, layers=4, fill=0.0): multi-layer glow + white core on any VMobject.\n"
        f"  * NeonDot(color=PK, radius=0.08): glowing point.\n"
        f"  * neon_trail(get_point, color=PK, time=0.9): glowing tail behind a moving point.\n"
        f"  * sym(s, color=CY, size=22): short symbols / numbers only (uses Text, NO LaTeX).\n"
        f"  * neon_box(label, color=CY, w=1.6, h=0.9, pos=ORIGIN): node with 1-3 char label.\n"
        f"  * curve(xs, ys): smooth VMobject from numpy arrays.\n"
        f"  * make_grid(): faint dark neon background grid.\n"
        f"  * send(a, b, color=GR, rt=0.8): glowing request packet traveling from a to b.\n"
        f"  * hud(fn, color=CY): live HUD stats in corner (symbols + numbers only).\n\n"
        f"HARD RULES (NEVER BREAK):\n"
        f"1. Output ONE class `class NeonScene(Scene):` with `construct(self)`.\n"
        f"2. ZERO LaTeX: NEVER use Tex, MathTex, DecimalNumber, Variable, axes.add_coordinates(). "
        f"Use sym() or Text for symbols (1-3 chars max, e.g. 'A', '7', 'x', 'f(x)').\n"
        f"3. ZERO descriptive sentences or titles on screen. Pure visual storytelling through shapes, glow, and motion.\n"
        f"4. EVERY visible line/curve/shape must be wrapped with neon(...). Every glowing point must be NeonDot.\n"
        f"5. Keep all objects inside safe frame: x in [-6.2, 6.2], y in [-3.3, 3.3].\n"
        f"6. Total animation duration must fit ~{duration:.1f}s. Every self.play(...) must have an explicit run_time.\n"
        f"7. Return ONLY the Python code inside ```python ... ``` block. No explanations."
    )
    try:
        log.info("Đang gọi AI chuyên code sinh Manim animation...")
        resp = generate_text(prompt, max_tokens=2048, task="code")
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
        log.warning("AI sinh code manim thất bại: %s", e)
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
import numpy as np

config.frame_rate = {fps}
config.pixel_width = {width}
config.pixel_height = {height}
config.background_color = "{bg}"
DURATION = {duration}

BG = "{bg}"
CY, PK, GR = "#00f0ff", "#ff2bd6", "#39ff14"
YE, PU, OR = "#ffe600", "#9d4dff", "#ff8a00"
FONT = "Monospace"

def neon(mob, color=CY, width=3, layers=4, fill=0.0):
    """Glow layers + colored line + white hot core. Works on any VMobject."""
    g = VGroup()
    for i in range(layers, 0, -1):
        g.add(mob.copy().set_fill(opacity=0)
                .set_stroke(color=color, width=width + i * 4,
                            opacity=0.04 + 0.025 * (layers - i)))
    g.add(mob.copy().set_fill(color=color, opacity=fill)
            .set_stroke(color=color, width=width, opacity=1))
    g.add(mob.copy().set_fill(opacity=0)
            .set_stroke(color="#ffffff", width=max(width * 0.35, 0.8), opacity=0.9))
    return g

class NeonDot(VGroup):
    """Glowing point. Move it with .move_to / .animate.move_to / updaters."""
    def __init__(self, color=PK, radius=0.08, layers=5, **kw):
        super().__init__(**kw)
        for i in range(layers, 0, -1):
            self.add(Circle(radius=radius + i * 0.07, stroke_width=0,
                            fill_color=color,
                            fill_opacity=0.04 + 0.02 * (layers - i)))
        self.add(Circle(radius=radius, stroke_width=0,
                        fill_color=color, fill_opacity=1))
        self.add(Circle(radius=radius * 0.45, stroke_width=0,
                        fill_color="#ffffff", fill_opacity=0.95))

def neon_trail(get_point, color=PK, time=0.9, width=4):
    """Short glowing tail behind a moving point."""
    return VGroup(
        TracedPath(get_point, stroke_color=color, stroke_width=width * 3,
                   stroke_opacity=0.12, dissipating_time=time),
        TracedPath(get_point, stroke_color=color, stroke_width=width * 1.6,
                   stroke_opacity=0.30, dissipating_time=time),
        TracedPath(get_point, stroke_color=color, stroke_width=width * 0.6,
                   stroke_opacity=1.0, dissipating_time=time))

def sym(s, color=CY, size=22):
    try:
        t = Text(str(s), font=FONT, font_size=size, color=color)
    except Exception:
        t = Text(str(s), font_size=size, color=color)
    t.set_stroke(color, width=4, opacity=0.25, background=True)
    return t

def neon_box(label, color=CY, w=1.6, h=0.9, pos=ORIGIN):
    box = neon(RoundedRectangle(corner_radius=0.2, width=w, height=h).move_to(pos),
               color, 3, 4, fill=0.12)
    return VGroup(box, sym(label, color, 22).move_to(pos))

def curve(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    m = VMobject()
    m.set_points_as_corners(np.c_[xs, ys, np.zeros_like(xs)])
    return m

def make_grid():
    return NumberPlane(
        background_line_style={"stroke_color": CY, "stroke_width": 1, "stroke_opacity": 0.08},
        axis_config={"stroke_color": CY, "stroke_width": 2, "stroke_opacity": 0.2})

def send(a, b, color=GR, rt=0.8):
    d = NeonDot(color, 0.07, 3).move_to(a)
    return Succession(FadeIn(d, run_time=0.05),
                      MoveAlongPath(d, Line(a, b), run_time=rt, rate_func=linear),
                      FadeOut(d, run_time=0.05))

def hud(fn, color=CY):
    return always_redraw(lambda: sym(fn(), color, 20).to_corner(UL, buff=0.4))

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
