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
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
try:
    import imageio_ffmpeg, matplotlib as _mpl
    _mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

# Dark Cyberpunk Styling Defaults
plt.rcParams['figure.facecolor'] = '#0a0d14'
plt.rcParams['axes.facecolor'] = '#0a0d14'
plt.rcParams['text.color'] = '#e6edf3'
plt.rcParams['axes.labelcolor'] = '#8b949e'
plt.rcParams['xtick.color'] = '#484f58'
plt.rcParams['ytick.color'] = '#484f58'
plt.rcParams['grid.color'] = '#161b22'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.5

# Cyberpunk Neon Palette Constants
CYAN = '#00f0ff'
NEON_GREEN = '#39ff14'
PINK = '#ff2bd6'
PURPLE = '#a855f7'
GOLD = '#ffe600'
DARK_BG = '#0a0d14'
CARD_BG = '#161b22'
BORDER_COLOR = '#30363d'

def glow_plot(ax, x, y, color=CYAN, lw=2.5, n_glow=3, **kw):
    """Vẽ đường cong phát sáng neon đa tầng (Cyberpunk Laser Glow)."""
    lines = []
    for i in range(n_glow, 0, -1):
        lines.append(ax.plot(x, y, color=color, lw=lw + i * 2.2, alpha=0.10 / i, **kw)[0])
    lines.append(ax.plot(x, y, color=color, lw=lw, alpha=0.95, **kw)[0])
    lines.append(ax.plot(x, y, color='#ffffff', lw=max(1.0, lw * 0.35), alpha=0.85, **kw)[0])
    return lines

def clean_axes(ax, keep_grid=True):
    """Loại bỏ viền thô, giữ giao diện sạch tối giản chuẩn tech studio."""
    for spine in ax.spines.values():
        spine.set_color('#30363d')
        spine.set_linewidth(1.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if keep_grid:
        ax.grid(True, linestyle='--', alpha=0.3, color='#21262d')

WIDTH = {width}
HEIGHT = {height}
FPS = {fps}
DURATION = {duration}
OUT_PATH = "out.mp4"
# MUST save: ani.save(OUT_PATH, fps=FPS, writer="ffmpeg", extra_args=['-pix_fmt', 'yuv420p', '-movflags', '+faststart'])
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


def _clean_and_validate_code(resp: str, framework: str = "matplotlib") -> str | None:
    """Tách block code Python sạch từ câu trả lời của AI và kiểm tra tính hợp lệ tối thiểu."""
    if not resp or not resp.strip():
        return None
    cleaned = resp.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        for i in range(1, len(parts), 2):
            block = parts[i]
            if block.startswith("python"):
                block = block[6:]
            c = block.strip()
            if c:
                cleaned = c
                break

    # Phát hiện nếu nội dung chỉ là thông báo lỗi API hoặc text từ chối
    lower = cleaned.lower()
    for err in (
        "enough credits",
        "paid pollen",
        "please top up",
        "credit balance",
        "insufficient credits",
        "insufficient_quota",
        "account behind this api key",
        "error 401",
        "error 403",
        "error 429",
    ):
        if err in lower:
            log.warning("Phản hồi AI không phải code mà là thông báo lỗi API: %s", cleaned[:150])
            return None

    # Kiểm tra cấu trúc tối thiểu của code theo framework
    if framework == "manim":
        if "Scene" not in cleaned and "construct" not in cleaned:
            log.warning("Code Manim thiếu cấu trúc class Scene / construct")
            return None
    elif framework == "matplotlib":
        if not any(k in cleaned for k in ("plt", "matplotlib", "FuncAnimation", "fig", "ax")):
            log.warning("Code Matplotlib không chứa các thành phần vẽ quen thuộc")
            return None

    return cleaned


def repair_code_with_ai(
    code: str,
    error_msg: str,
    framework: str = "matplotlib",
    duration: float = 5.0,
) -> str | None:
    """Gửi code bị lỗi và traceback cho LLM để tự động phân tích và sửa lỗi."""
    if not code or not code.strip():
        return None
    # Nếu code hiện tại vốn là thông báo lỗi API, không thử sửa vô ích
    if any(k in code.lower() for k in ("enough credits", "paid pollen", "top up", "insufficient credits")):
        return None

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
        return _clean_and_validate_code(resp, framework=framework)
    except Exception as e:
        log.warning("AI self-repair thất bại: %s", e)
        return None


def generate_matplotlib_from_prompt(
    description: str,
    duration: float = 5.0,
    topic: str = "",
    narration: str = "",
) -> str | None:
    """Yêu cầu AI viết code Matplotlib animation từ mô tả cảnh."""
    try:
        from .llm import generate_text
    except Exception as e:
        log.warning("Không thể import llm: %s", e)
        return None

    context_section = ""
    if topic or narration:
        context_section = (
            f"SCENE CONTEXT:\n"
            f"- Video Topic: {topic}\n"
            f"- Voiceover Narration: \"{narration}\"\n\n"
        )

    prompt = (
        f"You are a master mathematical visualization developer (3Blue1Brown & Kurzgesagt caliber).\n"
        f"Write an ULTRA-PRECISE, aesthetically gorgeous Python Matplotlib animation script for this concept:\n"
        f"{context_section}"
        f"VISUAL CONCEPT TO RENDER: {description}\n"
        f"TARGET DURATION: {duration:.1f} seconds.\n\n"
        f"PRECISION & CYBERPUNK VISUAL QUALITY GUIDELINES:\n"
        f"1. ACCURACY FIRST: Faithfully model the exact mechanism, mathematical formula, or algorithm progression.\n"
        f"   - Synchronize with the voiceover narration so the graphic visually matches what is being explained.\n"
        f"   - For data structures/algorithms: accurately simulate state changes (comparisons, pointers, swaps, traversal).\n"
        f"   - For math/physics/signals: plot exact functions, vectors, phase changes, or distributions.\n"
        f"2. PACING & RHYTHM (SLOW DOWN — NON-NEGOTIABLE):\n"
        f"   - SLOW DOWN: Each conceptual step MUST be visible for at least 0.8-1.5 seconds so the viewer can absorb it.\n"
        f"   - Max 3-4 key transitions for the full animation. Do NOT cram too many steps into a short video.\n"
        f"   - After each major state change or swap, insert a PAUSE (hold the frame steady for 0.5-1.0s).\n"
        f"   - Use np.linspace for gradual, smooth changes (smooth interpolation), NOT instant jarring jumps.\n"
        f"   - Spread the animation evenly over all frames = int(DURATION * FPS).\n"
        f"3. CYBERPUNK PALETTE & HELPERS (ALREADY INJECTED - USE THEM):\n"
        f"   - Palette constants available: CYAN ('#00f0ff'), NEON_GREEN ('#39ff14'), PINK ('#ff2bd6'), PURPLE ('#a855f7'), GOLD ('#ffe600'), DARK_BG ('#0a0d14').\n"
        f"   - Helper available: `glow_plot(ax, x, y, color=CYAN, lw=2.5)` creates multi-layer glowing neon laser curves! Use it for dynamic paths/data.\n"
        f"   - Helper available: `clean_axes(ax)` removes ugly borders, sets subtle grid.\n"
        f"4. CLEANLINESS: Minimalist styling, subtle grid, clear annotations (ax.annotate) for key points.\n\n"
        f"CRITICAL TECHNICAL RULES:\n"
        f"- The runner already injects: WIDTH, HEIGHT, FPS, DURATION, OUT_PATH = 'out.mp4', and sets dark background. DO NOT REDEFINE THEM.\n"
        f"- Use `import matplotlib.pyplot as plt` and `from matplotlib.animation import FuncAnimation`.\n"
        f"- Figure setup: `fig = plt.figure(figsize=(WIDTH/100, HEIGHT/100), dpi=100, facecolor=DARK_BG)`.\n"
        f"- Number of frames = int(DURATION * FPS).\n"
        f"- MUST save animation to OUT_PATH using `ani.save(OUT_PATH, fps=FPS, writer='ffmpeg', extra_args=['-pix_fmt', 'yuv420p', '-movflags', '+faststart'])`.\n"
        f"- Return ONLY Python code inside ```python ... ``` block. No markdown explanation outside."
    )
    try:
        log.info("Đang gọi AI chuyên code sinh Matplotlib animation mới...")
        resp = generate_text(prompt, max_tokens=2048, task="code")
        return _clean_and_validate_code(resp, framework="matplotlib")
    except Exception as e:
        log.warning("AI sinh code matplotlib thất bại: %s", e)
        return None


def generate_manim_from_prompt(
    description: str,
    duration: float = 5.0,
    topic: str = "",
    narration: str = "",
) -> str | None:
    """Yêu cầu AI viết code Manim animation từ mô tả cảnh / thuật toán visual."""
    try:
        from .llm import generate_text
    except Exception as e:
        log.warning("Không thể import llm: %s", e)
        return None

    context_section = ""
    if topic or narration:
        context_section = (
            f"SCENE CONTEXT:\n"
            f"- Video Topic: {topic}\n"
            f"- Voiceover Narration: \"{narration}\"\n\n"
        )

    prompt = (
        f"You are a master mathematical animator (like 3Blue1Brown / Grant Sanderson).\n"
        f"Build a SINGLE-FILE Manim (Community Edition) animation with an ultra-precise, gorgeous LIGHT NEON aesthetic.\n"
        f"The visual MUST be crystal-clear, pedagogically brilliant, and completely accurate to the concept.\n\n"
        f"{context_section}"
        f"VISUAL CONCEPT / ALGORITHM TO ANIMATE:\n"
        f"{description}\n\n"
        f"TARGET DURATION: {duration:.1f} seconds.\n\n"
        f"PEDAGOGICAL & VISUAL RULES FOR MAXIMUM PRECISION:\n"
        f"1. EXACT DYNAMIC MECHANISM: Faithfully depict the exact state transitions of the algorithm or mathematical concept.\n"
        f"   - Synchronize with the voiceover narration so the graphic visually matches what is being explained.\n"
        f"   - For data structures: clearly position nodes/elements, highlight active pointers, animate swaps/inserts/traversals.\n"
        f"   - For systems/networks: show sender, receiver, and data packets flowing across channels with send(a, b).\n"
        f"2. COLOR SEMANTICS (State-based contrast):\n"
        f"   - Base/Idle elements: CY ('#00f0ff') with subtle glow.\n"
        f"   - Currently inspecting/active pointer: YE ('#ffe600') or PK ('#ff2bd6').\n"
        f"   - Successful match / sorted / verified: GR ('#39ff14').\n"
        f"3. PACING & RHYTHM (SLOW DOWN — NON-NEGOTIABLE):\n"
        f"   - RHYTHM RULE (NON-NEGOTIABLE): Between EVERY self.play(), insert self.wait(0.6-1.0).\n"
        f"   - Max 4-5 play() calls for the full scene. Budget time carefully. Do NOT cram too many actions.\n"
        f"   - WRONG (too fast): self.play(A); self.play(B); self.play(C)\n"
        f"   - RIGHT: self.play(A, run_time=1.2); self.wait(0.8); self.play(B, run_time=1.0); self.wait(0.6);\n"
        f"   - Break the scene into 2-3 logical steps (e.g., Step 1: Reveal & Setup -> Step 2: Step-by-step Transformation -> Step 3: Result & Highlight).\n"
        f"   - Ensure total time (sum of self.play run_times + self.waits) matches ~{duration:.1f}s.\n"
        f"4. ENVIRONMENT & HELPERS (ALREADY INJECTED - DO NOT REDEFINE OR RE-IMPORT):\n"
        f"   - Manim, numpy, DURATION={duration:.1f}, BG='#05060f' are already loaded.\n"
        f"   - Palette: CY='#00f0ff', PK='#ff2bd6', GR='#39ff14', YE='#ffe600', PU='#9d4dff', OR='#ff8a00'.\n"
        f"   - Helpers available:\n"
        f"     * neon(mob, color=CY, width=3, layers=4, fill=0.0): multi-layer glow + white core on any VMobject.\n"
        f"     * NeonDot(color=PK, radius=0.08): glowing point.\n"
        f"     * neon_trail(get_point, color=PK, time=0.9): glowing tail behind a moving point.\n"
        f"     * sym(s, color=CY, size=22): short symbols / numbers only (Text-based, NO LaTeX).\n"
        f"     * neon_box(label, color=CY, w=1.6, h=0.9, pos=ORIGIN): node with 1-3 char label.\n"
        f"     * curve(xs, ys): smooth VMobject from numpy arrays.\n"
        f"     * make_grid(): faint dark neon background grid.\n"
        f"     * send(a, b, color=GR, rt=0.8): glowing request packet traveling from a to b.\n"
        f"     * hud(fn, color=CY): live HUD stats in corner (symbols + numbers only).\n\n"
        f"RECOMMENDED STRUCTURE PATTERN:\n"
        f"```python\n"
        f"class NeonScene(Scene):\n"
        f"    def construct(self):\n"
        f"        grid = make_grid()\n"
        f"        self.add(grid)\n"
        f"        # Phase 1: Setup elements\n"
        f"        box_a = neon_box('A', color=CY, pos=LEFT * 2.8)\n"
        f"        box_b = neon_box('B', color=CY, pos=RIGHT * 2.8)\n"
        f"        self.play(FadeIn(box_a), FadeIn(box_b), run_time=1.0)\n"
        f"        self.wait(0.5)\n"
        f"        # Phase 2: Action / Packet transfer / Computation\n"
        f"        self.play(send(box_a.get_center(), box_b.get_center(), color=GR, rt=1.2))\n"
        f"        self.wait(0.5)\n"
        f"        # Phase 3: Highlight state / Result\n"
        f"        self.play(box_b[0].animate.set_color(YE), run_time=0.8)\n"
        f"        remaining = max(DURATION - 4.0, 0.5)\n"
        f"        self.wait(remaining)\n"
        f"```\n\n"
        f"HARD CONSTRAINTS:\n"
        f"1. Output ONE class `class NeonScene(Scene):` with `construct(self)`.\n"
        f"2. ZERO LaTeX: NEVER use Tex, MathTex, DecimalNumber, Variable. Use sym() or Text for 1-4 character symbols/numbers.\n"
        f"3. ZERO long text or paragraphs on screen: pure visual storytelling through shapes, glow, packets, and motion.\n"
        f"4. Keep all objects inside safe frame: x in [-6.0, 6.0], y in [-3.2, 3.2].\n"
        f"5. Return ONLY Python code inside ```python ... ``` block. No explanations."
    )
    try:
        log.info("Đang gọi AI chuyên code sinh Manim animation...")
        resp = generate_text(prompt, max_tokens=2048, task="code")
        return _clean_and_validate_code(resp, framework="manim")
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
    timeout: int = 150,
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
        background_line_style={{"stroke_color": CY, "stroke_width": 1, "stroke_opacity": 0.08}},
        axis_config={{"stroke_color": CY, "stroke_width": 2, "stroke_opacity": 0.2}})

def send(a, b, color=GR, rt=0.8):
    d = NeonDot(color, 0.07, 3).move_to(a)
    return Succession(FadeIn(d, run_time=0.05),
                      MoveAlongPath(d, Line(a, b), run_time=rt, rate_func=linear),
                      FadeOut(d, run_time=0.05))

def hud(fn, color=CY):
    return always_redraw(lambda: sym(fn(), color, 20).to_corner(UL, buff=0.4))

# ================= CODE AI (MANIM) BÊN DƯỚI =================
'''


_MANIM_QUALITY_FLAG = {
    "low_quality": "l",
    "medium_quality": "m",
    "high_quality": "h",
    "production_quality": "p",
}


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
    stderr = ""
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
            vids = [
                p for p in out_dir.rglob("*.mp4")
                if p.stat().st_size > 1024 and "partial_movie_files" not in p.parts
            ]
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
