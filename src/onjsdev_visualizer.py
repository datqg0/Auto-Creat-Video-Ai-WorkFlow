"""Vertical algorithm visualizer for Shorts.

Replicates the signature aesthetic of YouTube Shorts channel @onjsdev:
- Aspect Ratio: 9:16 vertical (1080 x 1920)
- Theme: Sleek Developer Dark Mode (#090a0f, card container #12141e, border #232738)
- Top Section: Pill badges ([Algorithms], [Sorting], [O(n²)]), Bold Title, Live Metrics HUD
- Center Section: algorithm-specific animation (sorting bars, binary-search range,
  or graph/pathfinding grid).  It deliberately supports several algorithm families
  so a Shorts series does not silently collapse back to Bubble Sort.
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
from typing import Any

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
    sorted_indices: list[int] | None = None
    line_active: int = 1
    description: str = ""
    active_range: tuple[int, int] | None = None
    found_index: int | None = None


@dataclass
class GridStep:
    rows: int
    cols: int
    walls: set[tuple[int, int]]
    visited: set[tuple[int, int]]
    frontier: set[tuple[int, int]]
    path: list[tuple[int, int]]
    current: tuple[int, int] | None = None
    start: tuple[int, int] = (0, 0)
    goal: tuple[int, int] = (0, 0)
    line_active: int = 1
    description: str = ""


class AlgorithmSimulator:
    """Simulate algorithms and record compact states suitable for a Short."""

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

    @staticmethod
    def insertion_sort(initial: list[int]) -> list[SortStep]:
        arr = list(initial)
        steps = [SortStep(array=list(arr), sorted_indices=[0], line_active=1, description="Bắt đầu vùng đã sắp xếp")]
        for i in range(1, len(arr)):
            j = i
            steps.append(SortStep(array=list(arr), comparing=(j - 1, j), sorted_indices=list(range(i)), line_active=3, description=f"Chèn {arr[j]} vào đúng vị trí"))
            while j > 0 and arr[j - 1] > arr[j]:
                arr[j - 1], arr[j] = arr[j], arr[j - 1]
                steps.append(SortStep(array=list(arr), comparing=(j - 1, j), swapping=True, sorted_indices=list(range(i + 1)), line_active=5, description="Dịch phần tử lớn sang phải"))
                j -= 1
            steps.append(SortStep(array=list(arr), sorted_indices=list(range(i + 1)), line_active=6, description=f"Đã chèn xong vị trí {i}"))
        steps.append(SortStep(array=list(arr), sorted_indices=list(range(len(arr))), line_active=7, description="Insertion Sort hoàn tất!"))
        return steps

    @staticmethod
    def selection_sort(initial: list[int]) -> list[SortStep]:
        arr = list(initial)
        steps = [SortStep(array=list(arr), line_active=1, description="Tìm phần tử nhỏ nhất từng lượt")]
        for i in range(len(arr)):
            min_idx = i
            for j in range(i + 1, len(arr)):
                steps.append(SortStep(array=list(arr), comparing=(min_idx, j), sorted_indices=list(range(i)), line_active=4, description=f"So sánh min {arr[min_idx]} với {arr[j]}"))
                if arr[j] < arr[min_idx]:
                    min_idx = j
            if min_idx != i:
                arr[i], arr[min_idx] = arr[min_idx], arr[i]
                steps.append(SortStep(array=list(arr), comparing=(i, min_idx), swapping=True, sorted_indices=list(range(i)), line_active=6, description=f"Đưa {arr[i]} về đầu vùng chưa sắp xếp"))
            steps.append(SortStep(array=list(arr), sorted_indices=list(range(i + 1)), line_active=7, description=f"Chốt vị trí {i}"))
        return steps

    @staticmethod
    def merge_sort(initial: list[int]) -> list[SortStep]:
        arr = list(initial)
        steps = [SortStep(array=list(arr), line_active=1, description="Chia mảng thành các nửa nhỏ")]

        def _merge(lo: int, mid: int, hi: int) -> None:
            left, right = arr[lo:mid], arr[mid:hi]
            i = j = 0
            for k in range(lo, hi):
                li = lo + min(i, max(len(left) - 1, 0))
                rj = mid + min(j, max(len(right) - 1, 0))
                steps.append(SortStep(array=list(arr), comparing=(li, rj), line_active=5, description="Chọn phần tử nhỏ hơn từ hai nửa"))
                if j >= len(right) or (i < len(left) and left[i] <= right[j]):
                    arr[k] = left[i]
                    i += 1
                else:
                    arr[k] = right[j]
                    j += 1
                steps.append(SortStep(array=list(arr), comparing=(k, k), swapping=True, sorted_indices=list(range(lo, k + 1)), line_active=6, description=f"Ghi {arr[k]} vào vị trí {k}"))

        def _sort(lo: int, hi: int) -> None:
            if hi - lo <= 1:
                return
            mid = (lo + hi) // 2
            _sort(lo, mid)
            _sort(mid, hi)
            _merge(lo, mid, hi)

        _sort(0, len(arr))
        steps.append(SortStep(array=list(arr), sorted_indices=list(range(len(arr))), line_active=8, description="Merge Sort hoàn tất!"))
        return steps

    @staticmethod
    def heap_sort(initial: list[int]) -> list[SortStep]:
        arr = list(initial)
        n = len(arr)
        steps = [SortStep(array=list(arr), line_active=1, description="Dựng max-heap")]

        def _heapify(size: int, root: int) -> None:
            while True:
                largest, left, right = root, 2 * root + 1, 2 * root + 2
                if left < size:
                    steps.append(SortStep(array=list(arr), comparing=(largest, left), sorted_indices=list(range(size, n)), line_active=4, description="So sánh node với con trái"))
                    if arr[left] > arr[largest]:
                        largest = left
                if right < size:
                    steps.append(SortStep(array=list(arr), comparing=(largest, right), sorted_indices=list(range(size, n)), line_active=5, description="So sánh node với con phải"))
                    if arr[right] > arr[largest]:
                        largest = right
                if largest == root:
                    return
                arr[root], arr[largest] = arr[largest], arr[root]
                steps.append(SortStep(array=list(arr), comparing=(root, largest), swapping=True, sorted_indices=list(range(size, n)), line_active=6, description="Đẩy phần tử lớn lên đỉnh heap"))
                root = largest

        for root in range(n // 2 - 1, -1, -1):
            _heapify(n, root)
        for end in range(n - 1, 0, -1):
            arr[0], arr[end] = arr[end], arr[0]
            steps.append(SortStep(array=list(arr), comparing=(0, end), swapping=True, sorted_indices=list(range(end, n)), line_active=8, description=f"Chốt max ở vị trí {end}"))
            _heapify(end, 0)
        steps.append(SortStep(array=list(arr), sorted_indices=list(range(n)), line_active=9, description="Heap Sort hoàn tất!"))
        return steps

    @staticmethod
    def binary_search(initial: list[int], target: int | None = None) -> list[SortStep]:
        arr = sorted(initial)
        target = arr[len(arr) * 2 // 3] if target is None else target
        lo, hi = 0, len(arr) - 1
        steps = [SortStep(array=list(arr), active_range=(lo, hi), line_active=1, description=f"Tìm {target} trong mảng đã sắp xếp")]
        while lo <= hi:
            mid = (lo + hi) // 2
            steps.append(SortStep(array=list(arr), comparing=(mid, mid), active_range=(lo, hi), line_active=4, description=f"Kiểm tra phần tử giữa: {arr[mid]}"))
            if arr[mid] == target:
                steps.append(SortStep(array=list(arr), comparing=(mid, mid), active_range=(mid, mid), found_index=mid, line_active=5, description=f"Đã tìm thấy {target} tại vị trí {mid}!"))
                return steps
            if arr[mid] < target:
                lo = mid + 1
                description = "Bỏ nửa trái, tiếp tục bên phải"
                line = 7
            else:
                hi = mid - 1
                description = "Bỏ nửa phải, tiếp tục bên trái"
                line = 9
            steps.append(SortStep(array=list(arr), active_range=(lo, hi), line_active=line, description=description))
        steps.append(SortStep(array=list(arr), active_range=(0, -1), line_active=10, description=f"Không có {target} trong mảng"))
        return steps

    @staticmethod
    def pathfinding(kind: str = "bfs") -> list[GridStep]:
        rows, cols = 9, 8
        start, goal = (7, 0), (1, 7)
        walls = {(1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (5, 3), (5, 4), (3, 4), (2, 4), (2, 5), (2, 6), (6, 6), (7, 6)}
        frontier: list[tuple[int, int]] = [start]
        came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        distance: dict[tuple[int, int], int] = {start: 0}
        visited: set[tuple[int, int]] = set()
        steps = [GridStep(rows, cols, walls, set(), {start}, [], start, start, goal, 1, "Bắt đầu từ ô xanh")]

        def priority(cell: tuple[int, int]) -> tuple[int, int]:
            heuristic = abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])
            g_score = distance.get(cell, 10**9)
            if kind == "a_star":
                return (g_score + heuristic, heuristic)
            if kind == "dijkstra":
                return (g_score, heuristic)
            return (g_score, 0)

        while frontier:
            frontier.sort(key=priority)
            current = frontier.pop(0)
            visited.add(current)
            steps.append(GridStep(rows, cols, walls, set(visited), set(frontier), [], current, start, goal, 4, f"Mở rộng ô {current}"))
            if current == goal:
                break
            r, c = current
            for nxt in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                nr, nc = nxt
                if 0 <= nr < rows and 0 <= nc < cols and nxt not in walls and nxt not in came_from:
                    came_from[nxt] = current
                    distance[nxt] = distance[current] + 1
                    frontier.append(nxt)
            steps.append(GridStep(rows, cols, walls, set(visited), set(frontier), [], current, start, goal, 6, "Thêm các ô lân cận vào frontier"))

        path: list[tuple[int, int]] = []
        cur: tuple[int, int] | None = goal if goal in came_from else None
        while cur is not None:
            path.append(cur)
            cur = came_from[cur]
        path.reverse()
        steps.append(GridStep(rows, cols, walls, set(visited), set(), path, goal, start, goal, 8, f"Đường đi ngắn nhất có {max(0, len(path) - 1)} bước"))
        return steps


def render_onjsdev_frame(
    step: SortStep | GridStep,
    title: str = "BUBBLE SORT",
    category: str = "Algorithms",
    sub_category: str = "Sorting",
    complexity: str = "O(n²)",
    code_lines: list[str] | None = None,
    total_comparisons: int = 0,
    total_swaps: int = 0,
    visual_kind: str = "sort",
    target: int | None = None,
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
    if visual_kind == "grid" and isinstance(step, GridStep):
        hud_text = f"COMPLEXITY: {complexity}   |   VISITED: {len(step.visited):02d}   |   FRONTIER: {len(step.frontier):02d}"
    elif visual_kind == "search" and isinstance(step, SortStep):
        lo, hi = step.active_range or (0, len(step.array) - 1)
        hud_text = f"COMPLEXITY: {complexity}   |   CHECKS: {total_comparisons:02d}   |   RANGE: {max(0, hi - lo + 1):02d}"
    else:
        hud_text = f"COMPLEXITY: {complexity}   |   SWAPS: {total_swaps:02d}   |   COMPS: {total_comparisons:02d}"
    draw.text((70, 285), hud_text, font=font_hud, fill=COLOR_TEXT_MUTED)

    # 3. CENTER SECTION: Visualization Card (Y: 360 - 1280)
    card_x1, card_y1 = 60, 360
    card_x2, card_y2 = width - 60, 1280
    draw.rounded_rectangle([card_x1, card_y1, card_x2, card_y2], radius=24, fill=COLOR_CARD_BG, outline=COLOR_CARD_BORDER, width=2)

    if visual_kind == "grid" and isinstance(step, GridStep):
        grid_top = card_y1 + 120
        gap = 10
        cell = min(88, int(((card_x2 - card_x1) - 110 - gap * (step.cols - 1)) / step.cols))
        grid_w = step.cols * cell + (step.cols - 1) * gap
        grid_h = step.rows * cell + (step.rows - 1) * gap
        start_x = (width - grid_w) // 2
        start_y = grid_top + max(0, ((card_y2 - grid_top) - grid_h) // 2)
        for row in range(step.rows):
            for col in range(step.cols):
                cell_id = (row, col)
                x = start_x + col * (cell + gap)
                y = start_y + row * (cell + gap)
                fill = (28, 32, 46)
                outline = COLOR_CARD_BORDER
                if cell_id in step.walls:
                    fill, outline = (51, 65, 85), (71, 85, 105)
                elif cell_id in step.path:
                    fill, outline = COLOR_GREEN, COLOR_GREEN
                elif cell_id == step.current:
                    fill, outline = COLOR_AMBER, COLOR_AMBER
                elif cell_id in step.frontier:
                    fill, outline = (88, 28, 135), COLOR_PURPLE
                elif cell_id in step.visited:
                    fill, outline = (12, 82, 105), COLOR_CYAN
                if cell_id == step.start:
                    fill, outline = COLOR_CYAN, COLOR_CYAN
                elif cell_id == step.goal:
                    fill, outline = COLOR_CORAL, COLOR_CORAL
                draw.rounded_rectangle([x, y, x + cell, y + cell], radius=10, fill=fill, outline=outline, width=3)
                if cell_id == step.start or cell_id == step.goal:
                    label = "S" if cell_id == step.start else "G"
                    draw.text((x + cell // 2, y + cell // 2), label, font=font_num, fill=COLOR_TEXT_PRIMARY, anchor="mm")
    else:
        assert isinstance(step, SortStep)
        arr = step.array
        n = len(arr)
        avail_w = (card_x2 - card_x1) - 60
        base_y = card_y2 - 90
        active_lo, active_hi = step.active_range or (0, n - 1)
        sorted_set = set(step.sorted_indices or [])

        if visual_kind == "search":
            tile_gap = 12
            tile_w = min(105, int((avail_w - tile_gap * max(0, n - 1)) / max(n, 1)))
            total_w = n * tile_w + (n - 1) * tile_gap
            start_x = card_x1 + (card_x2 - card_x1 - total_w) // 2
            tile_y = card_y1 + 350
            draw.text((width // 2, card_y1 + 175), f"TARGET  {target}", font=_load_font(42, bold=True), fill=COLOR_CORAL, anchor="mm")
            for idx, val in enumerate(arr):
                x = start_x + idx * (tile_w + tile_gap)
                is_active = active_lo <= idx <= active_hi
                is_mid = bool(step.comparing and idx in step.comparing)
                color = COLOR_GREEN if idx == step.found_index else COLOR_AMBER if is_mid else COLOR_CYAN if is_active else (55, 65, 81)
                draw.rounded_rectangle([x, tile_y, x + tile_w, tile_y + tile_w], radius=14, fill=(20, 29, 45), outline=color, width=4 if (is_mid or idx == step.found_index) else 2)
                draw.text((x + tile_w // 2, tile_y + tile_w // 2), str(val), font=font_num, fill=color, anchor="mm")
                label = "MID" if is_mid else "L" if idx == active_lo else "R" if idx == active_hi else ""
                if label:
                    draw.text((x + tile_w // 2, tile_y + tile_w + 34), label, font=_load_font(20, bold=True), fill=color, anchor="mm")
            range_text = "FOUND" if step.found_index is not None else f"SEARCH RANGE  [{active_lo}..{active_hi}]"
            draw.text((width // 2, tile_y + tile_w + 120), range_text, font=_load_font(30, bold=True), fill=COLOR_GREEN if step.found_index is not None else COLOR_TEXT_MUTED, anchor="mm")
        else:
            max_val = max(arr) if arr else 1
            avail_h = (card_y2 - card_y1) - 160
            bar_w = max(24, int(avail_w / n) - 12)
            start_x = card_x1 + 30 + (avail_w - (n * (bar_w + 12))) // 2
            for idx, val in enumerate(arr):
                bx = start_x + idx * (bar_w + 12)
                bh = max(30, int((val / max_val) * avail_h))
                by = base_y - bh
                if step.swapping and step.comparing and idx in step.comparing:
                    bar_color = COLOR_CORAL
                elif step.comparing and idx in step.comparing:
                    bar_color = COLOR_AMBER
                elif idx in sorted_set:
                    bar_color = COLOR_GREEN
                else:
                    bar_color = COLOR_CYAN
                draw.rounded_rectangle([bx, by, bx + bar_w, base_y], radius=8, fill=bar_color)
                num_str = str(val)
                nb = font_num.getbbox(num_str)
                draw.text((bx + (bar_w - (nb[2] - nb[0])) // 2, by - 36), num_str, font=font_num, fill=bar_color)
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


SUPPORTED_ALGORITHMS = frozenset({
    "bubble_sort", "quick_sort", "merge_sort", "insertion_sort", "selection_sort", "heap_sort",
    "binary_search", "bfs", "dijkstra", "a_star",
})

_ALGORITHM_ALIASES = {
    "bubblesort": "bubble_sort", "bubble": "bubble_sort",
    "quicksort": "quick_sort", "quick": "quick_sort",
    "mergesort": "merge_sort", "merge": "merge_sort",
    "insertionsort": "insertion_sort", "insertion": "insertion_sort",
    "selectionsort": "selection_sort", "selection": "selection_sort",
    "heapsort": "heap_sort", "heap": "heap_sort",
    "binarysearch": "binary_search", "binary_search": "binary_search",
    "breadthfirstsearch": "bfs", "breadth_first_search": "bfs", "bfs": "bfs",
    "dijkstra": "dijkstra", "astar": "a_star", "a_star": "a_star", "a*": "a_star",
}


def normalize_algorithm_name(name: str | None) -> str | None:
    """Return a supported canonical algorithm name; never default to Bubble Sort."""
    raw = str(name or "").strip().lower().replace("-", "_").replace(" ", "_")
    compact = raw.replace("_", "")
    if raw in SUPPORTED_ALGORITHMS:
        return raw
    if raw in _ALGORITHM_ALIASES:
        return _ALGORITHM_ALIASES[raw]
    if compact in _ALGORITHM_ALIASES:
        return _ALGORITHM_ALIASES[compact]
    for alias in sorted(_ALGORITHM_ALIASES, key=len, reverse=True):
        if alias in raw or alias.replace("_", "") in compact:
            return _ALGORITHM_ALIASES[alias]
    return None


def generate_onjsdev_short_video(
    algorithm: str = "bubble_sort",
    initial_array: list[int] | None = None,
    output_path: Path | str = "output/onjsdev_short.mp4",
    fps: int = 30,
    steps_per_second: float = 2.5,
    target: int | None = None,
) -> Path:
    """Simulate one supported algorithm and compile its vertical animation to MP4."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if not initial_array:
        initial_array = [45, 12, 89, 34, 67, 23, 91, 55, 18, 76]
    initial_array = [int(v) for v in initial_array[:10]]
    canonical = normalize_algorithm_name(algorithm)
    if canonical is None:
        raise ValueError(f"Thuật toán chưa được visualizer hỗ trợ: {algorithm!r}")

    specs: dict[str, dict[str, Any]] = {
        "bubble_sort": {"title": "BUBBLE SORT", "complexity": "O(n²)", "sub": "Sorting", "kind": "sort", "simulate": lambda: AlgorithmSimulator.bubble_sort(initial_array), "code": None},
        "quick_sort": {"title": "QUICK SORT", "complexity": "O(n log n)", "sub": "Divide & Conquer", "kind": "sort", "simulate": lambda: AlgorithmSimulator.quick_sort(initial_array), "code": ["def quick_sort(a, lo, hi):", "    if lo >= hi: return", "    pivot = a[hi]", "    p = partition(a, pivot)", "    quick_sort(a, lo, p-1)", "    quick_sort(a, p+1, hi)"]},
        "merge_sort": {"title": "MERGE SORT", "complexity": "O(n log n)", "sub": "Divide & Conquer", "kind": "sort", "simulate": lambda: AlgorithmSimulator.merge_sort(initial_array), "code": ["def merge_sort(a):", "    if len(a) <= 1: return a", "    mid = len(a) // 2", "    left = merge_sort(a[:mid])", "    right = merge_sort(a[mid:])", "    return merge(left, right)"]},
        "insertion_sort": {"title": "INSERTION SORT", "complexity": "O(n²)", "sub": "Sorting", "kind": "sort", "simulate": lambda: AlgorithmSimulator.insertion_sort(initial_array), "code": ["for i in range(1, len(a)):", "    j = i", "    while j > 0:", "        if a[j-1] <= a[j]: break", "        swap(a, j-1, j)", "        j -= 1", "return a"]},
        "selection_sort": {"title": "SELECTION SORT", "complexity": "O(n²)", "sub": "Sorting", "kind": "sort", "simulate": lambda: AlgorithmSimulator.selection_sort(initial_array), "code": ["for i in range(len(a)):", "    min_i = i", "    for j in range(i+1, len(a)):", "        if a[j] < a[min_i]:", "            min_i = j", "    swap(a, i, min_i)", "return a"]},
        "heap_sort": {"title": "HEAP SORT", "complexity": "O(n log n)", "sub": "Heap", "kind": "sort", "simulate": lambda: AlgorithmSimulator.heap_sort(initial_array), "code": ["build_max_heap(a)", "for end in reversed(range(1, len(a))):", "    swap(a, 0, end)", "    heapify(a, root=0, size=end)", "return a"]},
        "binary_search": {"title": "BINARY SEARCH", "complexity": "O(log n)", "sub": "Search", "kind": "search", "simulate": lambda: AlgorithmSimulator.binary_search(initial_array, target), "code": ["lo, hi = 0, len(a) - 1", "while lo <= hi:", "    mid = (lo + hi) // 2", "    if a[mid] == target: return mid", "    if a[mid] < target:", "        lo = mid + 1", "    else: hi = mid - 1", "return -1"]},
        "bfs": {"title": "BREADTH-FIRST SEARCH", "complexity": "O(V + E)", "sub": "Pathfinding", "kind": "grid", "simulate": lambda: AlgorithmSimulator.pathfinding("bfs"), "code": ["queue = [start]", "while queue:", "    node = queue.pop(0)", "    if node == goal: break", "    for next in neighbors(node):", "        if next not in visited:", "            queue.append(next)", "return rebuild_path()"]},
        "dijkstra": {"title": "DIJKSTRA", "complexity": "O((V+E) log V)", "sub": "Shortest Path", "kind": "grid", "simulate": lambda: AlgorithmSimulator.pathfinding("dijkstra"), "code": ["dist[start] = 0", "heap = [(0, start)]", "while heap:", "    cost, node = heappop(heap)", "    if node == goal: break", "    for next in neighbors(node):", "        relax(node, next)", "return rebuild_path()"]},
        "a_star": {"title": "A* SEARCH", "complexity": "O((V+E) log V)", "sub": "Heuristic Search", "kind": "grid", "simulate": lambda: AlgorithmSimulator.pathfinding("a_star"), "code": ["open_set = [start]", "g[start] = 0", "while open_set:", "    node = min_f_score(open_set)", "    if node == goal: break", "    for next in neighbors(node):", "        relax_with_heuristic(next)", "return rebuild_path()"]},
    }
    spec = specs[canonical]
    title = str(spec["title"])
    complexity = str(spec["complexity"])
    sub_category = str(spec["sub"])
    visual_kind = str(spec["kind"])
    steps = spec["simulate"]()
    code_lines = spec["code"]
    effective_target = target
    if canonical == "binary_search" and effective_target is None:
        sorted_values = sorted(initial_array)
        effective_target = sorted_values[len(sorted_values) * 2 // 3]

    # Đặt frame tạm cạnh output: Windows Store/sandbox có thể trả về TEMP nhìn
    # thấy được nhưng không cho tiến trình con tạo thư mục bên trong.
    temp_dir = Path(tempfile.mkdtemp(prefix="onjsdev_render_", dir=str(out_file.parent.resolve())))
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
        if isinstance(st, SortStep) and st.swapping:
            total_swaps += 1
            audio_file = "swap.wav"
        elif isinstance(st, SortStep) and st.comparing:
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
            sub_category=sub_category,
            complexity=complexity,
            code_lines=code_lines,
            total_comparisons=total_comps,
            total_swaps=total_swaps,
            visual_kind=visual_kind,
            target=effective_target,
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
