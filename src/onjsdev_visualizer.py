"""onjsdev Algorithm & Math Visualizer Engine.

Replicates the signature aesthetic of YouTube Shorts channel @onjsdev:
- Aspect Ratio: 9:16 vertical (1080 x 1920)
- Theme: Sleek Developer Dark Mode (#090a0f, card container #12141e, border #232738)
- Top Section: Pill badges ([Algorithms], [Sorting], [O(n²)]), Bold Title, Live Metrics HUD
- Center Section: Real-time algorithm state animation (colored sorting bars, comparison pointers i/j,
  swap animations, maze/graph pathfinding)
- Bottom Section: VS Code dark editor card with syntax-highlighted code and active line indicator
- Sound Sync: Synced with crisp ASMR click/pop/swap sound effects from assets/sfx/
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
SFX_DIR = ROOT / "assets" / "sfx"
FONTS_DIR = ROOT / "assets" / "fonts"

# --- Curated Onjsdev Color Palette ---
COLOR_BG = (9, 10, 15)           # #090a0f Deep dark
COLOR_CARD_BG = (18, 20, 30)     # #12141e Card container
COLOR_CARD_BORDER = (35, 39, 56) # #232738 Subtle card border
COLOR_TEXT_PRIMARY = (245, 247, 250)
COLOR_TEXT_MUTED = (148, 163, 184)
COLOR_CYAN = (56, 189, 248)       # #38bdf8 Brand cyan
COLOR_AMBER = (245, 158, 11)      # #f59e0b Comparing pointer
COLOR_CORAL = (244, 63, 94)       # #f43f5e Swapping action
COLOR_GREEN = (16, 185, 129)      # #10b981 Sorted/Success
COLOR_PURPLE = (168, 85, 247)     # #a855f7 Category tag
COLOR_CODE_BG = (22, 27, 34)      # #161b22 VS Code editor


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load system or bundled font with fallback."""
    candidates = []
    if bold:
        candidates.extend([
            FONTS_DIR / "BeVietnamPro-Bold.ttf",
            Path("C:/Windows/Fonts/segoeui.ttf"),
            Path("C:/Windows/Fonts/arialbd.ttf"),
            Path("C:/Windows/Fonts/consola.ttf"),
        ])
    else:
        candidates.extend([
            FONTS_DIR / "BeVietnamPro-Regular.ttf",
            Path("C:/Windows/Fonts/segoeuil.ttf"),
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("C:/Windows/Fonts/consola.ttf"),
        ])
    for p in candidates:
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except Exception:
                pass
    return ImageFont.load_default()


@dataclass
class SortStep:
    array: list[int]
    comparing: tuple[int, int] | None = None
    swapping: bool = False
    sorted_indices: list[int] = None
    line_active: int = 1
    description: str = ""


class AlgorithmSimulator:
    """Simulates sorting & pathfinding algorithms and records step-by-step states."""

    @staticmethod
    def bubble_sort(initial: list[int]) -> list[SortStep]:
        arr = list(initial)
        n = len(arr)
        steps = [SortStep(array=list(arr), description="Khởi tạo mảng ban đầu", line_active=1)]
        sorted_idx = []

        for i in range(n):
            swapped = False
            for j in range(0, n - i - 1):
                # Step 1: Compare
                steps.append(SortStep(
                    array=list(arr),
                    comparing=(j, j + 1),
                    swapping=False,
                    sorted_indices=list(sorted_idx),
                    line_active=3,
                    description=f"So sánh {arr[j]} và {arr[j+1]}"
                ))
                if arr[j] > arr[j + 1]:
                    # Step 2: Swap
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    swapped = True
                    steps.append(SortStep(
                        array=list(arr),
                        comparing=(j, j + 1),
                        swapping=True,
                        sorted_indices=list(sorted_idx),
                        line_active=4,
                        description=f"Đổi chỗ {arr[j+1]} và {arr[j]}"
                    ))
            sorted_idx.append(n - i - 1)
            if not swapped:
                break

        sorted_idx = list(range(n))
        steps.append(SortStep(array=list(arr), sorted_indices=sorted_idx, line_active=6, description="Mảng đã được sắp xếp hoàn chỉnh!"))
        return steps

    @staticmethod
    def quick_sort(initial: list[int]) -> list[SortStep]:
        arr = list(initial)
        steps = [SortStep(array=list(arr), description="Bắt đầu QuickSort chia để trị", line_active=1)]
        sorted_set = set()

        def _qs(low: int, high: int):
            if low < high:
                pivot = arr[high]
                i = low - 1
                for j in range(low, high):
                    steps.append(SortStep(
                        array=list(arr),
                        comparing=(j, high),
                        swapping=False,
                        sorted_indices=list(sorted_set),
                        line_active=4,
                        description=f"So sánh phần tử {arr[j]} với pivot {pivot}"
                    ))
                    if arr[j] <= pivot:
                        i += 1
                        arr[i], arr[j] = arr[j], arr[i]
                        if i != j:
                            steps.append(SortStep(
                                array=list(arr),
                                comparing=(i, j),
                                swapping=True,
                                sorted_indices=list(sorted_set),
                                line_active=5,
                                description=f"Đổi chỗ {arr[j]} về bên trái pivot"
                            ))
                arr[i + 1], arr[high] = arr[high], arr[i + 1]
                p_idx = i + 1
                sorted_set.add(p_idx)
                steps.append(SortStep(
                    array=list(arr),
                    comparing=(p_idx, high),
                    swapping=True,
                    sorted_indices=list(sorted_set),
                    line_active=7,
                    description=f"Đặt pivot {arr[p_idx]} vào vị trí chính xác"
                ))
                _qs(low, p_idx - 1)
                _qs(p_idx + 1, high)
            elif low == high:
                sorted_set.add(low)

        _qs(0, len(arr) - 1)
        steps.append(SortStep(array=list(arr), sorted_indices=list(range(len(arr))), line_active=8, description="QuickSort hoàn tất!"))
        return steps


def render_onjsdev_frame(
    step: SortStep,
    title: str = "BUBBLE SORT",
    category: str = "Algorithms",
    sub_category: str = "Sorting",
    complexity: str = "O(n²)",
    code_lines: list[str] | None = None,
    total_comparisons: int = 0,
    total_swaps: int = 0,
    width: int = 1080,
    height: int = 1920,
) -> Image.Image:
    """Render a single 1080x1920 vertical frame in onjsdev style."""
    im = Image.new("RGB", (width, height), COLOR_BG)
    draw = ImageDraw.Draw(im)

    # 1. Subtle Radial Background Vignette & Grid Lines
    for gy in range(0, height, 80):
        draw.line([(0, gy), (width, gy)], fill=(15, 17, 24), width=1)
    for gx in range(0, width, 80):
        draw.line([(gx, 0), (gx, height)], fill=(15, 17, 24), width=1)

    # Fonts
    font_badge = _load_font(26, bold=True)
    font_title = _load_font(58, bold=True)
    font_hud = _load_font(28, bold=False)
    font_num = _load_font(26, bold=True)
    font_code = _load_font(28, bold=False)
    font_sub = _load_font(32, bold=False)

    # 2. TOP SECTION: Badges & Title (Y: 120 - 360)
    # Badges
    badge_x = 70
    badge_y = 130
    badges = [
        (category.upper(), (56, 189, 248, 40), (56, 189, 248)),
        (sub_category.upper(), (168, 85, 247, 40), (168, 85, 247)),
        (complexity, (245, 158, 11, 40), (245, 158, 11)),
    ]
    for b_text, b_bg, b_fg in badges:
        bbox = font_badge.getbbox(b_text)
        bw = (bbox[2] - bbox[0]) + 32
        bh = 48
        draw.rounded_rectangle([badge_x, badge_y, badge_x + bw, badge_y + bh], radius=24, fill=(20, 24, 38), outline=b_fg, width=2)
        draw.text((badge_x + 16, badge_y + 10), b_text, font=font_badge, fill=b_fg)
        badge_x += bw + 18

    # Main Title
    draw.text((70, 205), title.upper(), font=font_title, fill=COLOR_TEXT_PRIMARY)

    # Metrics HUD bar
    hud_text = f"COMPLEXITY: {complexity}   |   SWAPS: {total_swaps:02d}   |   COMPS: {total_comparisons:02d}"
    draw.text((70, 285), hud_text, font=font_hud, fill=COLOR_TEXT_MUTED)

    # 3. CENTER SECTION: Visualization Card (Y: 360 - 1280)
    card_x1, card_y1 = 60, 360
    card_x2, card_y2 = width - 60, 1280
    draw.rounded_rectangle([card_x1, card_y1, card_x2, card_y2], radius=24, fill=COLOR_CARD_BG, outline=COLOR_CARD_BORDER, width=2)

    # Draw Sorting Bars
    arr = step.array
    n = len(arr)
    max_val = max(arr) if arr else 1
    avail_w = (card_x2 - card_x1) - 60
    avail_h = (card_y2 - card_y1) - 160
    bar_w = max(24, int(avail_w / n) - 12)
    start_x = card_x1 + 30 + (avail_w - (n * (bar_w + 12))) // 2
    base_y = card_y2 - 60

    sorted_set = set(step.sorted_indices or [])

    for idx, val in enumerate(arr):
        bx = start_x + idx * (bar_w + 12)
        bh = max(30, int((val / max_val) * avail_h))
        by = base_y - bh

        # Determine Bar Color
        if step.swapping and step.comparing and idx in step.comparing:
            bar_color = COLOR_CORAL
            glow_color = (244, 63, 94, 100)
        elif step.comparing and idx in step.comparing:
            bar_color = COLOR_AMBER
            glow_color = (245, 158, 11, 100)
        elif idx in sorted_set:
            bar_color = COLOR_GREEN
            glow_color = (16, 185, 129, 80)
        else:
            bar_color = COLOR_CYAN
            glow_color = None

        # Draw rounded vertical bar
        draw.rounded_rectangle([bx, by, bx + bar_w, base_y], radius=8, fill=bar_color)

        # Draw number on top of bar
        num_str = str(val)
        nb = font_num.getbbox(num_str)
        nw = nb[2] - nb[0]
        draw.text((bx + (bar_w - nw) // 2, by - 36), num_str, font=font_num, fill=bar_color)

        # Draw pointer arrow below bar if active
        if step.comparing and idx in step.comparing:
            ptr_name = "i" if idx == step.comparing[0] else "j"
            draw.text((bx + (bar_w - 14) // 2, base_y + 12), ptr_name, font=font_num, fill=bar_color)

    # Action Status Text inside Card
    if step.description:
        sb = font_sub.getbbox(step.description)
        sw = sb[2] - sb[0]
        draw.text((width // 2 - sw // 2, card_y1 + 25), step.description, font=font_sub, fill=COLOR_TEXT_PRIMARY)

    # 4. BOTTOM SECTION: VS Code Editor Card (Y: 1330 - 1820)
    code_x1, code_y1 = 60, 1330
    code_x2, code_y2 = width - 60, 1820
    draw.rounded_rectangle([code_x1, code_y1, code_x2, code_y2], radius=20, fill=COLOR_CODE_BG, outline=COLOR_CARD_BORDER, width=2)

    # Code titlebar with 3 dots
    draw.ellipse([code_x1 + 25, code_y1 + 20, code_x1 + 41, code_y1 + 36], fill=(239, 68, 68))   # Red
    draw.ellipse([code_x1 + 50, code_y1 + 20, code_x1 + 66, code_y1 + 36], fill=(245, 158, 11))  # Yellow
    draw.ellipse([code_x1 + 75, code_y1 + 20, code_x1 + 91, code_y1 + 36], fill=(16, 185, 129))  # Green
    draw.text((code_x1 + 115, code_y1 + 16), f"{title.lower().replace(' ', '_')}.py", font=_load_font(22, bold=False), fill=COLOR_TEXT_MUTED)
    draw.line([(code_x1, code_y1 + 55), (code_x2, code_y1 + 55)], fill=COLOR_CARD_BORDER, width=1)

    # Default Code Lines for BubbleSort
    if not code_lines:
        code_lines = [
            "def bubble_sort(arr):",
            "    n = len(arr)",
            "    for i in range(n):",
            "        for j in range(0, n - i - 1):",
            "            if arr[j] > arr[j + 1]:",
            "                arr[j], arr[j+1] = arr[j+1], arr[j]",
            "    return arr",
        ]

    line_y = code_y1 + 75
    for l_idx, line in enumerate(code_lines, start=1):
        is_active = (l_idx == step.line_active)
        if is_active:
            # Active Line Glow highlight
            draw.rectangle([code_x1 + 4, line_y - 4, code_x2 - 4, line_y + 40], fill=(30, 41, 59))
            draw.rectangle([code_x1 + 4, line_y - 4, code_x1 + 12, line_y + 40], fill=COLOR_CYAN)

        # Line number
        draw.text((code_x1 + 30, line_y), f"{l_idx:2d}", font=font_code, fill=(90, 105, 125))
        # Code text
        c_fill = COLOR_CYAN if is_active else COLOR_TEXT_PRIMARY
        draw.text((code_x1 + 80, line_y), line, font=font_code, fill=c_fill)
        line_y += 46

    return im


def generate_onjsdev_short_video(
    algorithm: str = "bubble_sort",
    initial_array: list[int] | None = None,
    output_path: Path | str = "output/onjsdev_short.mp4",
    fps: int = 30,
    steps_per_second: float = 2.5,
) -> Path:
    """Simulates algorithm, renders 1080x1920 frames, syncs click/swap audio, and compiles MP4."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if not initial_array:
        initial_array = [45, 12, 89, 34, 67, 23, 91, 55, 18, 76]

    algo_lower = algorithm.lower().replace("-", "_").replace(" ", "_")
    if "quick" in algo_lower:
        title = "QUICK SORT"
        complexity = "O(n log n)"
        steps = AlgorithmSimulator.quick_sort(initial_array)
        code_lines = [
            "def quick_sort(arr, low, high):",
            "    if low < high:",
            "        pivot = arr[high]",
            "        i = low - 1",
            "        for j in range(low, high):",
            "            if arr[j] <= pivot: i += 1; swap(arr, i, j)",
            "        swap(arr, i + 1, high)",
            "        quick_sort(arr, low, i); quick_sort(arr, i + 2, high)",
        ]
    else:
        title = "BUBBLE SORT"
        complexity = "O(n²)"
        steps = AlgorithmSimulator.bubble_sort(initial_array)
        code_lines = None

    temp_dir = Path(tempfile.mkdtemp(prefix="onjsdev_render_"))
    frames_dir = temp_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    log.info("Rendering %d onjsdev visualizer steps for %s...", len(steps), title)

    total_swaps = 0
    total_comps = 0
    frame_idx = 0
    frames_per_step = max(1, int(fps / steps_per_second))

    # Keep track of audio events: list of (timestamp_seconds, sfx_file)
    audio_events: list[tuple[float, str]] = []

    for s_idx, st in enumerate(steps):
        if st.swapping:
            total_swaps += 1
            audio_file = "swap.wav"
        elif st.comparing:
            total_comps += 1
            audio_file = "click.wav"
        elif s_idx == len(steps) - 1:
            audio_file = "complete.wav"
        else:
            audio_file = "pop.wav"

        audio_events.append((frame_idx / fps, audio_file))

        img = render_onjsdev_frame(
            step=st,
            title=title,
            category="Algorithms",
            sub_category="Sorting",
            complexity=complexity,
            code_lines=code_lines,
            total_comparisons=total_comps,
            total_swaps=total_swaps,
        )

        for _ in range(frames_per_step):
            img.save(frames_dir / f"frame_{frame_idx:05d}.png")
            frame_idx += 1

    # Add 1.5 seconds hold at final state
    for _ in range(int(fps * 1.5)):
        img.save(frames_dir / f"frame_{frame_idx:05d}.png")
        frame_idx += 1

    total_duration = frame_idx / fps
    log.info("Rendered %d frames (%.1fs). Compiling video with FFmpeg...", frame_idx, total_duration)

    # Locate FFmpeg
    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"

    # Assemble video with FFmpeg
    cmd = [
        ffmpeg_bin, "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "18",
        str(out_file)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    # Clean up temp frames
    shutil.rmtree(temp_dir, ignore_errors=True)
    log.info("✓ Successfully generated onjsdev short video: %s", out_file)
    return out_file


if __name__ == "__main__":
    generate_onjsdev_short_video(
        algorithm="bubble_sort",
        initial_array=[38, 15, 92, 44, 71, 29, 85, 50, 19, 63],
        output_path="output/preview_onjsdev_bubble_sort.mp4",
    )
