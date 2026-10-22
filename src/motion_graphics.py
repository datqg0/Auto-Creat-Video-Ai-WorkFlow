"""Motion Graphics Engine dựng bằng HTML5 Canvas + Headless Browser (Edge/Chrome) + FFmpeg.

Phong cách Tech Studio / Cyberpunk Minimalist. Thu frame tất định (deterministic) rồi ghép MP4 H.264.

Nguyên tắc v3:
  * DATA-DRIVEN: mọi template chỉ vẽ dữ liệu LẤY TỪ Scene (bullets, chart, animation, narration).
    Không còn số liệu bịa cứng (12,500 ms, 97%, 67k stars...). Phần dữ liệu minh hoạ tự sinh
    (trọng số attention, mảng mặc định...) luôn kèm huy hiệu "MINH HOẠ" / "ILLUSTRATIVE".
  * ROUTING có cấu trúc: dựa visual_type / animation.preset / chart / algorithm trước,
    từ khoá (có ranh giới từ regex) chỉ là tín hiệu phụ, chấm điểm rồi chọn template có dữ liệu.
  * Template/JS nằm trong chuỗi thường (không f-string), cấu hình đưa vào 1 object JSON an toàn.
  * Runner bền vững: profile tạm độc lập, server đa luồng, đủ 100% frame mới nhận, kill cả cây process.

Template: conclusion, quote_card, terminal, diff_view, comparison, bar_benchmark, github_stats (đường tăng trưởng),
hud (bộ đếm), scheduler (interval scheduling), cpu_scheduler, stickman, git_flow, attention_map,
neural_flow, algo_trace, metaphor, architecture (fallback: các bước tuần tự).

Khoá CONFIG tuỳ chọn (CONFIG["motion_graphics"]): browser_path, font_file, keep_frames, min_timeout,
extra_args (list). Biến môi trường MOTION_BROWSER cũng ghi đè đường dẫn trình duyệt.
"""
from __future__ import annotations

import base64
import http.server
import json
import logging
import math
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .config import CONFIG
from .models import Scene

log = logging.getLogger(__name__)

__all__ = [
    "get_browser_path",
    "get_edge_path",
    "get_ffmpeg_path",
    "is_motion_graphics_available",
    "render_motion_graphic_scene",
    "render_preview_png",
]

_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Trình duyệt headless & FFmpeg
# ---------------------------------------------------------------------------


def _mg_cfg() -> dict:
    cfg = CONFIG.get("motion_graphics", {}) if hasattr(CONFIG, "get") else {}
    return cfg if isinstance(cfg, dict) else {}


def get_ffmpeg_path() -> str | None:
    """Tìm đường dẫn FFmpeg thực thi (hỗ trợ Hermes tools, imageio, PATH, cấu hình)."""
    for forced in (os.environ.get("FFMPEG_PATH"), _mg_cfg().get("ffmpeg_path")):
        if forced and os.path.exists(str(forced)):
            return str(forced)

    found = shutil.which("ffmpeg")
    if found:
        return found

    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    candidates = [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
    ]
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        hermes_dir = Path(local_app_data) / "hermes" / "tools"
        if hermes_dir.exists():
            for p in hermes_dir.glob("**/ffmpeg.exe"):
                if p.is_file():
                    return str(p)

    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def get_browser_path() -> str | None:
    """Tìm Edge/Chrome/Chromium trên Windows / Linux / macOS (ưu tiên cấu hình/biến môi trường)."""
    for forced in (os.environ.get("MOTION_BROWSER"), _mg_cfg().get("browser_path")):
        if forced and os.path.exists(str(forced)):
            return str(forced)

    for name in ("msedge", "microsoft-edge", "google-chrome", "google-chrome-stable",
                 "chromium", "chromium-browser", "chrome"):
        found = shutil.which(name)
        if found:
            return found

    candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Edge\Application\msedge.exe"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/usr/bin/google-chrome", "/usr/bin/google-chrome-stable", "/usr/bin/chromium",
        "/usr/bin/chromium-browser", "/usr/bin/microsoft-edge", "/snap/bin/chromium",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def get_edge_path() -> str | None:
    """Tương thích ngược cho module cũ."""
    return get_browser_path()


def is_motion_graphics_available() -> bool:
    return get_browser_path() is not None


# ---------------------------------------------------------------------------
# Nhãn UI theo ngôn ngữ
# ---------------------------------------------------------------------------

_LABELS = {
    "vi": {
        "illustrative": "MINH HOẠ",
        "quote_tag": "BẢN CHẤT CỐT LÕI",
        "problem": "VẤN ĐỀ", "solution": "GIẢI PHÁP",
        "room": "Phòng", "rooms_needed": "Cần tối thiểu {n} phòng", "conflict": "TRÙNG LỊCH",
        "compared": "So sánh", "swaps": "Hoán đổi", "step": "Bước",
        "running": "Đang chạy", "done": "Hoàn thành", "queue": "HÀNG ĐỢI", "total": "Tổng",
        "ratio": "Chênh lệch gấp {x} lần", "old": "PHIÊN BẢN CŨ", "new": "PHIÊN BẢN MỚI",
        "key": "KEY", "query": "QUERY", "input": "Đầu vào", "output": "Đầu ra", "hidden": "Ẩn",
        "commits": ["Khởi tạo", "Tính năng", "Sửa lỗi", "Hợp nhất", "Phát hành"],
        "default_title": "PHÂN TÍCH KỸ THUẬT", "branch": "nhánh",
        "calc_step": "Phép tính", "result_cell": "Kết quả", "scalar_mult": "Nhân vô hướng",
    },
    "en": {
        "illustrative": "ILLUSTRATIVE",
        "quote_tag": "CORE IDEA",
        "problem": "PROBLEM", "solution": "SOLUTION",
        "room": "Room", "rooms_needed": "At least {n} rooms needed", "conflict": "CONFLICT",
        "compared": "Comparisons", "swaps": "Swaps", "step": "Step",
        "running": "Running", "done": "Done", "queue": "QUEUE", "total": "Total",
        "ratio": "{x}x difference", "old": "OLD VERSION", "new": "NEW VERSION",
        "key": "KEY", "query": "QUERY", "input": "Input", "output": "Output", "hidden": "Hidden",
        "commits": ["Init", "Feature", "Bugfix", "Merge", "Release"],
        "default_title": "TECHNICAL BREAKDOWN", "branch": "branch",
        "calc_step": "Calculation", "result_cell": "Result", "scalar_mult": "Scalar Multiplication",
    },
}

# ---------------------------------------------------------------------------
# Tiện ích văn bản / trích dữ liệu từ Scene
# ---------------------------------------------------------------------------

_HEADING_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"\[\s*(?:hook|intro|outro|cảnh|scene|phần|part|bước|step|pha|chương|chapter)\s*\d*\s*\]\s*[:\-–—|]?"
    r"|(?:hook|intro|outro|cảnh|scene|phần|part|bước|step|pha|chương|chapter|mở đầu)\s*\d*\s*[:\-–—|]"
    r")\s*",
    re.IGNORECASE,
)
_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")
_EMOJI_RE = re.compile("[\U00010000-\U0010ffff\u2600-\u27bf\u200d\ufe0f]")


def _clean_title(text: str | None, lang: str = "vi") -> str:
    """Gỡ tiền tố 'Hook:', 'Cảnh 1 -'... (chỉ khi có dấu phân cách, không cắt 'Bước 3 là...')."""
    t = _EMOJI_RE.sub("", (text or "")).strip()
    for _ in range(3):
        stripped = _HEADING_PREFIX_RE.sub("", t, count=1).strip()
        if stripped == t:
            break
        t = stripped
    t = re.sub(r"\s+", " ", t).strip(" :-–—|")
    return t or _LABELS.get(lang, _LABELS["vi"])["default_title"]


def _truncate(text: str, n: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= n:
        return text
    cut = text[: n - 1]
    if " " in cut[int(n * 0.5):]:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:-–—") + "…"


def _sentences(text: str | None, n: int, max_chars: int = 120) -> list[str]:
    parts = [p.strip() for p in _SENT_SPLIT.split(_EMOJI_RE.sub("", text or "")) if p.strip()]
    return [_truncate(p, max_chars) for p in parts[:n]]


def _g(scene: Scene, name: str, default: Any = None) -> Any:
    v = getattr(scene, name, default)
    return default if v is None else v


def _anim(scene: Scene) -> dict:
    a = _g(scene, "animation", {})
    return a if isinstance(a, dict) else {}


def _bullets(scene: Scene, limit: int = 12) -> list[str]:
    out = []
    for b in _g(scene, "bullets", []) or []:
        s = re.sub(r"\s+", " ", _EMOJI_RE.sub("", str(b))).strip()
        if s:
            out.append(s)
    return out[:limit]


def _to_float(s: str) -> float | None:
    s = (s or "").strip().replace(" ", "")
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s):
        s = s.replace(",", "")
    elif re.fullmatch(r"-?\d{1,3}(\.\d{3})+(,\d+)?", s):
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")
    try:
        v = float(s)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def _num(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v) if math.isfinite(float(v)) else None
    if isinstance(v, str):
        return _to_float(v)
    return None


_LV_RE = re.compile(r"^\s*(?P<label>[^:=]+?)\s*[:=]\s*(?P<val>-?\d[\d.,]*)\s*(?P<unit>\S.*)?$")


def _label_values(bullets: list[str]) -> list[tuple[str, float, str, str]]:
    """'Brute force: 12.5 ms' -> (label, value, unit, raw_value_text)."""
    out = []
    for b in bullets:
        m = _LV_RE.match(b)
        if not m:
            continue
        v = _to_float(m.group("val"))
        if v is None:
            continue
        out.append((m.group("label").strip(), v, (m.group("unit") or "").strip(),
                    (m.group("val") + " " + (m.group("unit") or "")).strip()))
    return out


# ---------------------------------------------------------------------------
# Data builders: trả dict (có thể kèm "_illustrative": True) hoặc None nếu scene không đủ dữ liệu
# ---------------------------------------------------------------------------

def _steps(scene: Scene, lang: str) -> list[str]:
    steps = _bullets(scene, 5)
    if not steps:
        steps = _sentences(scene.narration, 4, 90)
    if not steps:
        steps = [_clean_title(scene.heading, lang)]
    return [_truncate(s, 110) for s in steps]


def _d_takeaways(scene: Scene, lang: str) -> dict | None:
    items = _bullets(scene, 4) or _sentences(scene.narration, 3, 150)
    return {"takeaways": [_truncate(i, 150) for i in items]} if items else None


def _d_quote(scene: Scene, lang: str) -> dict | None:
    b = _bullets(scene, 1)
    q = b[0] if b else (_clean_title(scene.heading, lang) if scene.heading else "")
    if not q:
        s = _sentences(scene.narration, 1, 200)
        q = s[0] if s else ""
    if q and q.strip().lower() in ("outro", "outro.", "kết thúc"):
        q = "Cảm ơn bạn đã theo dõi! Hãy Like & Đăng ký kênh để đón xem các tập công nghệ tiếp theo."
    return {"quote": q} if q else None


def _chart(scene: Scene) -> dict | None:
    c = _g(scene, "chart", None)
    if not isinstance(c, dict):
        return None
    labels, values = c.get("labels"), c.get("values")
    if not isinstance(labels, list) or not isinstance(values, list) or len(labels) != len(values) or len(values) < 2:
        return None
    nums = [_num(v) for v in values]
    if any(v is None for v in nums):
        return None
    return {"kind": c.get("kind", "bar"), "labels": [str(x) for x in labels], "values": nums,
            "unit": str(c.get("unit", "") or "")}


def _d_bars(scene: Scene, lang: str) -> dict | None:
    c, a = _chart(scene), _anim(scene)
    labels = values = None
    unit = ""
    if c and c["kind"] in ("bar", "pie"):
        labels, values, unit = c["labels"], c["values"], c["unit"]
    elif a.get("preset") == "bar_chart" and isinstance(a.get("values"), list):
        nums = [_num(v) for v in a["values"]]
        if len(nums) >= 2 and all(v is not None for v in nums):
            values = nums
            raw = a.get("labels") if isinstance(a.get("labels"), list) else []
            labels = [str(raw[i]) if i < len(raw) else str(i + 1) for i in range(len(nums))]
    if values is None:
        lv = _label_values(_bullets(scene))
        if len(lv) >= 2:
            labels, values = [x[0] for x in lv], [x[1] for x in lv]
            units = {x[2] for x in lv}
            unit = units.pop() if len(units) == 1 else ""
    if values is None:
        return None
    return {"bars": {"labels": [_truncate(x, 40) for x in labels[:8]], "values": values[:8], "unit": unit}}


def _d_series(scene: Scene, lang: str) -> dict | None:
    c = _chart(scene)
    if not c or c["kind"] != "line":
        return None
    return {"series": {"labels": [_truncate(x, 16) for x in c["labels"][:12]], "values": c["values"][:12], "unit": c["unit"]}}


def _d_hud(scene: Scene, lang: str) -> dict | None:
    a = _anim(scene)
    if a.get("preset") != "counter":
        return None
    v = _num(a.get("to_value"))
    if v is None:
        return None
    metrics = [{"label": _truncate(l, 18), "value": _truncate(raw, 14)} for l, _v, _u, raw in _label_values(_bullets(scene))[:3]]
    return {"hud": {"value": v, "unit": str(a.get("unit", "") or ""), "metrics": metrics}}


def _d_code(scene: Scene, lang: str) -> dict | None:
    raw = [str(b).replace("\t", "    ").rstrip() for b in (_g(scene, "bullets", []) or [])]
    lines = [ln for ln in raw if ln.strip()][:12]
    if not lines:
        return None
    return {"code": {"lines": lines, "lang": str(_g(scene, "code_language", "") or "")}}


def _d_diff(scene: Scene, lang: str) -> dict | None:
    raw = [str(b).rstrip() for b in (_g(scene, "bullets", []) or []) if str(b).strip()]
    old = [ln[1:].strip() for ln in raw if ln.lstrip().startswith("-") and not ln.lstrip().startswith("--")]
    new = [ln[1:].strip() for ln in raw if ln.lstrip().startswith("+")]
    if old and new:
        return {"diff": {"old": old[:8], "new": new[:8]}}
    if len(raw) >= 2:
        k = math.ceil(len(raw) / 2)
        return {"diff": {"old": raw[:k][:8], "new": raw[k:][:8]}}
    return None


def _d_compare(scene: Scene, lang: str) -> dict | None:
    b = _bullets(scene, 10)
    if len(b) < 2:
        return None
    k = math.ceil(len(b) / 2)
    left, right = b[:k], b[k:]

    def parse_side(items: list[str]) -> dict:
        if not items:
            return {"title": "", "rows": []}
        first = items[0]
        if ":" in first:
            t, r = first.split(":", 1)
            rows = ([r.strip()] if r.strip() else []) + [x for x in items[1:]]
            return {"title": _truncate(t.strip(), 30), "rows": [_truncate(x, 90) for x in rows[:4]]}
        return {"title": _truncate(first, 30), "rows": [_truncate(x, 90) for x in items[1:5]]}

    return {"cmp": {"left": parse_side(left), "right": parse_side(right)}}


_INTERVAL_RE = re.compile(
    r"^(?P<label>.*?)[\s:,(\[]*(?P<h1>\d{1,2})(?:[:h](?P<m1>\d{2})|h)\s*(?:[-–—]|->|→|đến|to)\s*"
    r"(?P<h2>\d{1,2})(?:[:h](?P<m2>\d{2})|h)[\s)\]]*$",
    re.IGNORECASE,
)


def _assign_rooms(ivs: list[dict]) -> int:
    """Interval partitioning tham lam (sắp theo giờ bắt đầu, first-fit) => số phòng tối thiểu."""
    order = sorted(range(len(ivs)), key=lambda i: (ivs[i]["start"], ivs[i]["end"]))
    ends: list[int] = []
    for i in order:
        for r, e in enumerate(ends):
            if e <= ivs[i]["start"]:
                ends[r] = ivs[i]["end"]
                ivs[i]["room"] = r
                break
        else:
            ends.append(ivs[i]["end"])
            ivs[i]["room"] = len(ends) - 1
    return len(ends)


def _d_intervals(scene: Scene, lang: str) -> dict | None:
    ivs: list[dict] = []
    for b in _bullets(scene, 8):
        m = _INTERVAL_RE.match(b)
        if not m:
            continue
        s = int(m["h1"]) * 60 + int(m["m1"] or 0)
        e = int(m["h2"]) * 60 + int(m["m2"] or 0)
        if int(m["h1"]) > 24 or int(m["h2"]) > 24 or e <= s:
            continue
        ivs.append({"label": _truncate(m["label"].strip(" -–—:,") or f"#{len(ivs) + 1}", 30), "start": s, "end": e})
    if len(ivs) < 2:
        return None
    rooms = _assign_rooms(ivs)
    return {"intervals": ivs, "rooms": rooms}


_TASK_RE = re.compile(r"^(?P<name>.+?)\s*[:(\-–]\s*(?P<d>\d+(?:[.,]\d+)?)\s*(?P<u>ms|µs|us|s|giây|ns)?\)?$", re.IGNORECASE)


def _d_tasks(scene: Scene, lang: str) -> dict | None:
    b = _bullets(scene, 6)
    if len(b) < 2:
        return None
    tasks = []
    for item in b:
        m = _TASK_RE.match(item)
        if m and _to_float(m["d"]) is not None:
            tasks.append({"name": _truncate(m["name"], 22), "dur": _to_float(m["d"]), "unit": m["u"] or ""})
        else:
            tasks.append({"name": _truncate(item, 22), "dur": None, "unit": ""})
    return {"tasks": tasks}


def _d_layers(scene: Scene, lang: str) -> dict | None:
    a = _anim(scene)
    layers = a.get("layers")
    if isinstance(layers, list):
        nums = [int(n) for n in (_num(x) for x in layers) if n is not None and n >= 1][:6]
        if len(nums) >= 2:
            return {"layers": [min(n, 8) for n in nums]}
    m = re.search(r"(?<!\d)(\d{1,2})(?:\s*[-–→,]\s*(\d{1,2})){2,5}(?!\d)", " ".join(_bullets(scene)))
    if m:
        nums = [int(x) for x in re.findall(r"\d{1,2}", m.group(0))]
        if len(nums) >= 3:
            return {"layers": [min(max(n, 1), 8) for n in nums[:6]]}
    if str(_g(scene, "algorithm", "")).lower() == "neural_network" or a.get("preset") == "neural_net":
        return {"layers": [3, 5, 4, 2], "_illustrative": True}
    return None


_ARRAY_RE = re.compile(r"\[\s*-?\d+(?:\s*,\s*-?\d+){2,}\s*\]")
_SORT_KEYS = (
    ("bubble", ("bubble", "nổi bọt")), ("insertion", ("insertion", "chèn")),
    ("selection", ("selection", "chọn")), ("quick", ("quicksort", "quick sort", "sắp xếp nhanh")),
)


def _d_array(scene: Scene, lang: str) -> dict | None:
    a = _anim(scene)
    arr = None
    for key in ("data", "values"):
        if a.get("preset") == "sorting" and isinstance(a.get(key), list):
            nums = [_num(x) for x in a[key]]
            if len(nums) >= 3 and all(x is not None for x in nums):
                arr = [int(x) for x in nums]
                break
    if arr is None:
        for src in (*_bullets(scene), scene.visual_prompt or "", scene.narration or "", scene.heading or ""):
            m = _ARRAY_RE.search(src)
            if m:
                arr = [int(x) for x in re.findall(r"-?\d+", m.group(0))]
                break
    illustrative = False
    if arr is None:
        if str(_g(scene, "algorithm", "")).lower() == "sorting" or a.get("preset") == "sorting":
            arr, illustrative = [5, 2, 8, 1, 9, 3], True
        else:
            return None
    arr = arr[:12]
    corpus = f"{scene.heading or ''} {scene.visual_prompt or ''} {(scene.narration or '')[:300]}".lower()
    kind = next((k for k, kws in _SORT_KEYS if any(w in corpus for w in kws)), None)
    if kind is None:
        kind, illustrative = "bubble", True
    return {"array": arr, "algo": kind, "_illustrative": illustrative}


def _d_attention(scene: Scene, lang: str) -> dict | None:
    toks = [_truncate(t, 10) for t in _bullets(scene, 7)]
    if len(toks) < 3:
        words = re.findall(r"[\wÀ-ỹ]+", (_sentences(scene.narration, 1, 200) or [scene.heading or ""])[0])
        toks = words[:6]
    if len(toks) < 3:
        toks = ["The", "cat", "sat", "on", "mat"]
    return {"tokens": toks[:7], "_illustrative": True}  # trọng số luôn là mô phỏng


def _d_git(scene: Scene, lang: str) -> dict | None:
    b = _bullets(scene, 5)
    if len(b) >= 2:
        return {"commits": [_truncate(x, 28) for x in b]}
    return {"commits": _LABELS[lang]["commits"][:4], "_illustrative": True}


def _d_metaphor(scene: Scene, lang: str, has_image: bool) -> dict | None:
    caps = [_truncate(x, 60) for x in _bullets(scene, 3)]
    if not has_image and not caps:
        return None
    return {"captions": caps}


def _d_matrix_op(scene: Scene, lang: str) -> dict | None:
    a = _anim(scene)
    preset = a.get("preset")
    is_explicit = preset in ("matrix_op", "matrix")
    corpus = f"{scene.heading or ''} {scene.visual_prompt or ''} {scene.narration or ''} {' '.join(_bullets(scene))}".lower()
    has_matrix_kw = any(w in corpus for w in ("ma trận", "matrix", "scalar", "vô hướng", "linear algebra", "đại số tuyến tính", "tensor"))

    matrix_a = a.get("matrix") or a.get("matrix_a")
    scalar = a.get("scalar")
    op = a.get("op", "×")

    # Try parsing matrix from bullets if not in animation
    if not matrix_a:
        found_rows = []
        for b in _bullets(scene):
            m = re.findall(r"-?\d+", b)
            if len(m) >= 2:
                found_rows.append([int(x) for x in m[:4]])
        if len(found_rows) >= 2:
            matrix_a = found_rows[:3]

    if scalar is None:
        for b in _bullets(scene):
            m = re.search(r"(?:scalar|c|k|hệ số|vô hướng)\s*[:=]?\s*(-?\d+)", b, re.IGNORECASE)
            if m:
                scalar = int(m.group(1))
                break

    illustrative = False
    if not matrix_a and not has_matrix_kw:
        return None

    if not matrix_a:
        matrix_a = [[2, 5, 1], [4, 0, 3]]
        scalar = 3 if scalar is None else scalar
        illustrative = True

    if scalar is None:
        scalar = 3

    rows = len(matrix_a)
    cols = max(len(r) for r in matrix_a)
    matrix_a = [[r[c] if c < len(r) else 0 for c in range(cols)] for r in matrix_a]

    result_matrix = []
    steps = []
    for r in range(rows):
        res_row = []
        for c in range(cols):
            val_a = matrix_a[r][c]
            res = val_a * scalar
            res_row.append(res)
            steps.append({
                "row": r,
                "col": c,
                "expr": f"{val_a} × {scalar} = {res}",
                "val": res,
            })
        result_matrix.append(res_row)

    bc = a.get("breadcrumb") or (f"{scene.heading.upper()} / MATRIX OP" if scene.heading else "LINEAR ALGEBRA / MATRIX MULTIPLICATION")
    diff = a.get("difficulty") or "Easy"
    cat = a.get("category") or "Linear Algebra"

    return {
        "matrix_op": {
            "scalar": scalar,
            "op": op,
            "matrix_a": matrix_a,
            "result_matrix": result_matrix,
            "rows": rows,
            "cols": cols,
            "steps": steps,
            "breadcrumb": bc,
            "difficulty": diff,
            "category": cat,
            "formula_label": f"Scalar Multiplication: C[i, j] = {scalar} · A[i, j]",
        },
        "_illustrative": illustrative,
    }


def _d_visual_algo(scene: Scene, lang: str) -> dict | None:
    a = _anim(scene)
    preset = a.get("preset")
    is_explicit = preset in ("visual_algo", "algo_table", "table_step")
    corpus = f"{scene.heading or ''} {scene.visual_prompt or ''} {scene.narration or ''} {' '.join(_bullets(scene))}".lower()
    has_algo_kw = any(w in corpus for w in ("covariance", "hiệp phương sai", "thống kê", "deep-ml", "worked example", "bảng tính", "step by step", "do they move together"))

    table = a.get("table")
    formula = a.get("formula")

    illustrative = False
    if not table and not has_algo_kw:
        return None

    if not table:
        table = {
            "headers": ["Day", "X (Sunshine)", "Y (Ice Cream)", "X - X̄", "Y - Ȳ", "Product"],
            "rows": [
                ["1", "2", "4", "-2", "-2", "4"],
                ["2", "4", "6", " 0", " 0", "0"],
                ["3", "6", "8", "+2", "+2", "4"],
            ],
        }
        formula = formula or "Cov(X, Y) = Σ [(X - X̄)(Y - Ȳ)] / (N - 1) = 8 / 2 = 4"
        illustrative = True

    bc = a.get("breadcrumb") or (f"{scene.heading.upper()} / WORKED EXAMPLE" if scene.heading else "STATISTICS / STEP BY STEP")
    diff = a.get("difficulty") or "Easy"
    cat = a.get("category") or "Statistics"

    return {
        "visual_algo": {
            "table": table,
            "formula": formula or "",
            "breadcrumb": bc,
            "difficulty": diff,
            "category": cat,
        },
        "_illustrative": illustrative,
    }


def _d_tree_graph(scene: Scene, lang: str) -> dict | None:
    a = _anim(scene)
    preset = a.get("preset")
    is_explicit = preset in ("tree_graph", "tree", "bst", "graph")
    corpus = f"{scene.heading or ''} {scene.visual_prompt or ''} {scene.narration or ''} {' '.join(_bullets(scene))}".lower()
    has_kw = any(w in corpus for w in ("cây nhị phân", "bst", "binary search tree", "binary tree", "cây", "tree", "đồ thị", "graph", "dfs", "bfs", "duyệt cây"))

    has_tree_data = bool(a.get("nodes"))
    if not has_tree_data and not has_kw:
        return None

    target = a.get("target") or 12
    nodes = a.get("nodes") or [
        {"id": 1, "val": 10, "level": 0, "pos": 0},
        {"id": 2, "val": 5, "level": 1, "pos": -1, "parent": 1},
        {"id": 3, "val": 15, "level": 1, "pos": 1, "parent": 1},
        {"id": 4, "val": 2, "level": 2, "pos": -2, "parent": 2},
        {"id": 5, "val": 7, "level": 2, "pos": -1, "parent": 2},
        {"id": 6, "val": 12, "level": 2, "pos": 1, "parent": 3},
        {"id": 7, "val": 20, "level": 2, "pos": 2, "parent": 3},
    ]
    path = a.get("path") or [10, 15, 12]

    bc = a.get("breadcrumb") or (f"{scene.heading.upper()} / TREE GRAPH" if scene.heading else "DATA STRUCTURES / BINARY TREE")
    diff = a.get("difficulty") or "Medium"
    cat = a.get("category") or "Data Structures"

    return {
        "tree_graph": {
            "target": target,
            "nodes": nodes,
            "path": path,
            "breadcrumb": bc,
            "difficulty": diff,
            "category": cat,
        },
        "_illustrative": not bool(a.get("nodes")),
    }


def _d_two_pointers(scene: Scene, lang: str) -> dict | None:
    a = _anim(scene)
    preset = a.get("preset")
    is_explicit = preset in ("two_pointers", "sliding_window", "two_pointer")
    corpus = f"{scene.heading or ''} {scene.visual_prompt or ''} {scene.narration or ''} {' '.join(_bullets(scene))}".lower()
    has_kw = any(w in corpus for w in ("hai con trỏ", "two pointer", "two pointers", "sliding window", "cửa sổ trượt", "con trỏ", "left pointer", "right pointer"))

    if not is_explicit and not has_kw:
        return None

    data = a.get("data")
    if not isinstance(data, list):
        data = [2, 3, 5, 7, 11, 15]
        illustrative = True
    else:
        illustrative = False

    target = a.get("target") or 16
    steps = a.get("steps") or [
        {"left": 0, "right": 5, "sum": 17, "action": "17 > 16 -> Move R Left (R--)"},
        {"left": 0, "right": 4, "sum": 13, "action": "13 < 16 -> Move L Right (L++)"},
        {"left": 1, "right": 4, "sum": 14, "action": "14 < 16 -> Move L Right (L++)"},
        {"left": 2, "right": 4, "sum": 16, "action": "5 + 11 = 16 == TARGET (MATCH FOUND!)", "matched": True},
    ]

    bc = a.get("breadcrumb") or "DEEP-ML / PROBLEM 03 / TWO POINTERS"
    diff = a.get("difficulty") or "Easy"
    cat = a.get("category") or "Algorithms"

    return {
        "two_pointers": {
            "data": data,
            "target": target,
            "steps": steps,
            "breadcrumb": bc,
            "difficulty": diff,
            "category": cat,
        },
        "_illustrative": illustrative,
    }


def _d_stack_queue(scene: Scene, lang: str) -> dict | None:
    a = _anim(scene)
    preset = a.get("preset")
    is_explicit = preset in ("stack_queue", "stack", "queue")
    corpus = f"{scene.heading or ''} {scene.visual_prompt or ''} {scene.narration or ''} {' '.join(_bullets(scene))}".lower()
    has_kw = any(w in corpus for w in ("ngăn xếp", "stack", "hàng đợi", "queue", "lifo", "fifo", "push", "pop"))

    if not is_explicit and not has_kw:
        return None

    kind = a.get("kind") or ("queue" if "queue" in corpus or "hàng đợi" in corpus else "stack")
    ops = a.get("ops") or [
        {"op": "PUSH", "val": 42},
        {"op": "PUSH", "val": 17},
        {"op": "PUSH", "val": 99},
        {"op": "POP", "val": 99},
        {"op": "PUSH", "val": 55},
    ]

    bc = a.get("breadcrumb") or (f"DEEP-ML / PROBLEM 04 / {kind.upper()} ({'LIFO' if kind == 'stack' else 'FIFO'})")
    diff = a.get("difficulty") or "Easy"
    cat = a.get("category") or "Data Structures"

    return {
        "stack_queue": {
            "kind": kind,
            "ops": ops,
            "breadcrumb": bc,
            "difficulty": diff,
            "category": cat,
        },
        "_illustrative": not bool(a.get("ops")),
    }


def _d_dp_grid(scene: Scene, lang: str) -> dict | None:
    a = _anim(scene)
    preset = a.get("preset")
    is_explicit = preset in ("dp_grid", "dp", "dynamic_programming")
    corpus = f"{scene.heading or ''} {scene.visual_prompt or ''} {scene.narration or ''} {' '.join(_bullets(scene))}".lower()
    has_kw = any(w in corpus for w in ("quy hoạch động", "dynamic programming", "dp table", "bảng dp", "lcs", "knapsack", "subproblem"))

    if not is_explicit and not has_kw:
        return None

    rows = a.get("rows") or 3
    cols = a.get("cols") or 4
    formula = a.get("formula") or "dp[i][j] = dp[i-1][j] + dp[i][j-1]"

    bc = a.get("breadcrumb") or "DEEP-ML / PROBLEM 12 / DYNAMIC PROGRAMMING"
    diff = a.get("difficulty") or "Medium"
    cat = a.get("category") or "Algorithms"

    return {
        "dp_grid": {
            "rows": rows,
            "cols": cols,
            "formula": formula,
            "row_headers": a.get("row_headers") or [f"i={r}" for r in range(rows)],
            "col_headers": a.get("col_headers") or [f"j={c}" for c in range(cols)],
            "breadcrumb": bc,
            "difficulty": diff,
            "category": cat,
        },
        "_illustrative": not bool(a.get("grid")),
    }


def _d_flowchart(scene: Scene, lang: str) -> dict | None:
    vt = str(_g(scene, "visual_type", "")).lower()
    a = _anim(scene)
    preset = a.get("preset")
    is_explicit = preset in ("flowchart", "flowchart_diagram", "pipeline", "pipeline_flow")
    corpus = f"{scene.heading or ''} {scene.visual_prompt or ''} {scene.narration or ''} {' '.join(_bullets(scene))}".lower()
    has_kw = any(w in corpus for w in ("sơ đồ", "flowchart", "pipeline", "luồng xử lý", "luồng dữ liệu", "quy trình", "data flow", "xử lý"))

    if not is_explicit and vt != "diagram" and not has_kw:
        return None

    nodes = a.get("nodes")
    connections = a.get("connections")
    bullets = _bullets(scene, 6)

    if not nodes and bullets:
        nodes = [{"id": i, "label": _truncate(b, 35)} for i, b in enumerate(bullets[:5])]
        connections = [[i, i + 1] for i in range(len(nodes) - 1)]

    if not nodes:
        nodes = [
            {"id": 0, "label": "Client Request"},
            {"id": 1, "label": "API Gateway"},
            {"id": 2, "label": "Event Queue"},
            {"id": 3, "label": "Worker Service"},
            {"id": 4, "label": "Database Cache"},
        ]
        connections = [[0, 1], [1, 2], [2, 3], [3, 4]]

    bc = a.get("breadcrumb") or "SYSTEM ARCHITECTURE / PIPELINE / DATA FLOW"
    diff = a.get("difficulty") or "System Design"
    cat = a.get("category") or "Architecture"

    return {
        "flowchart": {
            "nodes": nodes,
            "connections": connections or [[i, i + 1] for i in range(len(nodes) - 1)],
            "breadcrumb": bc,
            "difficulty": diff,
            "category": cat,
        },
        "_illustrative": not bool(a.get("nodes")),
    }


def _d_system_layers(scene: Scene, lang: str) -> dict | None:
    a = _anim(scene)
    preset = a.get("preset")
    is_explicit = preset in ("system_layers", "architecture_layers", "layers", "layer_stack")
    corpus = f"{scene.heading or ''} {scene.visual_prompt or ''} {scene.narration or ''} {' '.join(_bullets(scene))}".lower()
    has_kw = any(w in corpus for w in ("phân tầng", "system layers", "kiến trúc phân tầng", "multi-tier", "clean architecture", "tầng kiến trúc", "layer", "tier"))

    has_layer_data = bool(a.get("layers"))
    if not has_layer_data and not has_kw and len(_bullets(scene, 4)) < 3:
        return None

    layers = a.get("layers")
    if not layers:
        bullets = _bullets(scene, 4)
        if len(bullets) >= 3:
            layers = [{"name": _truncate(b, 25), "components": ["Core Module", "Sub-system"]} for b in bullets]
        elif has_kw:
            layers = [
                {"name": "Client Layer", "components": ["Web SPA", "Mobile App", "CLI"]},
                {"name": "Gateway Tier", "components": ["Rate Limiter", "Auth Service", "Reverse Proxy"]},
                {"name": "Core Services", "components": ["Business Logic", "Event Bus", "Workers"]},
                {"name": "Data Storage", "components": ["Redis Cache", "PostgreSQL", "S3 Storage"]},
            ]
        else:
            return None

    bc = a.get("breadcrumb") or (f"{scene.heading.upper()} / ARCHITECTURE" if scene.heading else "SYSTEM ARCHITECTURE / MULTI-TIER LAYERS")
    diff = a.get("difficulty") or "Design Pattern"
    cat = a.get("category") or "System Design"

    return {
        "system_layers": {
            "layers": layers,
            "breadcrumb": bc,
            "difficulty": diff,
            "category": cat,
        },
        "_illustrative": not bool(a.get("layers")),
    }


def _d_comparison_matrix(scene: Scene, lang: str) -> dict | None:
    a = _anim(scene)
    preset = a.get("preset")
    is_explicit = preset in ("comparison_matrix", "feature_matrix", "benchmark_table", "table")
    corpus = f"{scene.heading or ''} {scene.visual_prompt or ''} {scene.narration or ''} {' '.join(_bullets(scene))}".lower()
    has_kw = any(w in corpus for w in ("bảng so sánh", "ma trận", "bảng biểu", "tiêu chí", "feature matrix", "so sánh chi tiết", "đặc tính kỹ thuật", "matrix table"))

    headers = a.get("headers") or (a.get("table", {}).get("headers") if isinstance(a.get("table"), dict) else None)
    rows = a.get("rows") or (a.get("table", {}).get("rows") if isinstance(a.get("table"), dict) else None)

    has_table_data = bool(headers and rows)
    colon_bullets = [b for b in _bullets(scene, 5) if ":" in b]
    if not has_table_data and not has_kw and len(colon_bullets) < 2:
        return None

    if not headers or not rows:
        bullets = _bullets(scene, 5)
        parsed_rows = []
        for b in bullets:
            if ":" in b:
                parts = b.split(":", 1)
                parsed_rows.append([parts[0].strip(), parts[1].strip(), "Tối ưu ✔"])
        if len(parsed_rows) >= 2:
            headers = ["Tiêu chí", "Phương pháp A", "Phương pháp B"]
            rows = parsed_rows
        elif has_kw:
            headers = ["Tiêu chí", "Giải pháp Cũ (Naive)", "Giải pháp Mới (Modern)"]
            rows = [
                ["Độ phức tạp thời gian", "O(N²)", "O(N log N) ✔"],
                ["Bộ nhớ phụ trợ", "O(N)", "O(1) ✔"],
                ["Khả năng mở rộng (Scale)", "Kém ✖", "Cực tốt ✔"],
                ["Độ trễ phản hồi (P99)", "450 ms", "12 ms ✔"],
            ]
        else:
            return None

    bc = a.get("breadcrumb") or (f"{scene.heading.upper()} / COMPARISON" if scene.heading else "COMPARISON MATRIX / TECHNICAL BENCHMARK")
    diff = a.get("difficulty") or "Trade-off"
    cat = a.get("category") or "Analysis"

    return {
        "comparison_matrix": {
            "headers": headers,
            "rows": rows,
            "breadcrumb": bc,
            "difficulty": diff,
            "category": cat,
        },
        "_illustrative": not bool(a.get("rows")),
    }


def _d_radar_chart(scene: Scene, lang: str) -> dict | None:
    a = _anim(scene)
    preset = a.get("preset")
    chart = _chart(scene)
    chart_kind = chart.get("kind") if chart else None
    is_explicit = preset in ("radar_chart", "radar", "spider_chart") or chart_kind == "radar"
    corpus = f"{scene.heading or ''} {scene.visual_prompt or ''} {scene.narration or ''} {' '.join(_bullets(scene))}".lower()
    has_kw = any(w in corpus for w in ("biểu đồ radar", "radar chart", "mạng nhện", "đa chiều", "trade-off", "đánh đổi đa chiều", "spider chart"))

    if not is_explicit and not has_kw:
        return None

    axes = a.get("axes") or ["Tốc độ", "Bộ nhớ", "Mở rộng", "Tin cậy", "Chi phí"]
    series = a.get("series") or [
        {"name": "Giải pháp A", "values": [40, 80, 45, 85, 90], "color": "#ff2bd6"},
        {"name": "Giải pháp B", "values": [95, 85, 90, 70, 60], "color": "#00f0ff"},
    ]

    bc = a.get("breadcrumb") or "RADAR CHART / MULTI-CRITERIA EVALUATION"
    diff = a.get("difficulty") or "Trade-off"
    cat = a.get("category") or "Evaluation"

    return {
        "radar_chart": {
            "axes": axes,
            "series": series,
            "breadcrumb": bc,
            "difficulty": diff,
            "category": cat,
        },
        "_illustrative": not bool(a.get("series")),
    }


def _d_trie_tree(scene: Scene, lang: str) -> dict | None:
    a = _anim(scene)
    preset = a.get("preset")
    is_explicit = preset in ("trie_tree", "trie", "prefix_tree")
    corpus = f"{scene.heading or ''} {scene.visual_prompt or ''} {scene.narration or ''} {' '.join(_bullets(scene))}".lower()
    has_kw = any(w in corpus for w in ("trie", "prefix tree", "cây tiền tố", "autocomplete", "chuỗi ký tự"))

    if not is_explicit and not has_kw:
        return None

    words = a.get("words") or ["cat", "car", "cart", "dog"]
    query = a.get("query") or "car"
    bc = a.get("breadcrumb") or "DATA STRUCTURES / TRIE / PREFIX TREE"
    diff = a.get("difficulty") or "Medium"
    cat = a.get("category") or "String Algorithms"

    return {
        "trie_tree": {
            "words": words,
            "query": query,
            "breadcrumb": bc,
            "difficulty": diff,
            "category": cat,
        },
        "_illustrative": not bool(a.get("words")),
    }


def _d_graph_network(scene: Scene, lang: str) -> dict | None:
    a = _anim(scene)
    preset = a.get("preset")
    is_explicit = preset in ("graph_network", "network_graph", "dijkstra", "shortest_path")
    corpus = f"{scene.heading or ''} {scene.visual_prompt or ''} {scene.narration or ''} {' '.join(_bullets(scene))}".lower()
    has_kw = any(w in corpus for w in ("dijkstra", "shortest path", "đường đi ngắn nhất", "đồ thị có trọng số", "weighted graph", "network topology"))

    if not is_explicit and not has_kw:
        return None

    raw_nodes = a.get("nodes") or ["A", "B", "C", "D", "E"]
    nodes = []
    for n in raw_nodes:
        if isinstance(n, dict):
            nid = str(n.get("id") or n.get("name") or n.get("label") or "Node")
            lbl = str(n.get("label") or nid)
            nodes.append({"id": nid, "label": lbl})
        else:
            nodes.append({"id": str(n), "label": str(n)})

    raw_edges = a.get("edges") or [["A", "B", 4], ["A", "C", 2], ["C", "B", 1], ["B", "D", 5], ["C", "D", 8], ["D", "E", 3]]
    edges = []
    for e in raw_edges:
        if isinstance(e, dict):
            u = str(e.get("from") or e.get("u") or e.get("source") or "")
            v = str(e.get("to") or e.get("v") or e.get("target") or "")
            w = e.get("weight") if e.get("weight") is not None else (e.get("w") if e.get("w") is not None else 1)
            if u and v:
                edges.append([u, v, w])
        elif isinstance(e, (list, tuple)) and len(e) >= 2:
            w = e[2] if len(e) >= 3 else 1
            edges.append([str(e[0]), str(e[1]), w])

    raw_path = a.get("path") or ["A", "C", "B", "D", "E"]
    path = [str(p.get("id") or "") if isinstance(p, dict) else str(p) for p in raw_path]
    is_dijkstra = any(w in corpus for w in ("dijkstra", "shortest path", "đường đi ngắn nhất"))
    default_bc = "GRAPH THEORY / DIJKSTRA / SHORTEST PATH" if is_dijkstra else ((scene.heading or "NETWORK & GRAPH TOPOLOGY").upper())
    bc = a.get("breadcrumb") or default_bc
    diff = a.get("difficulty") or ("Medium" if is_dijkstra else "Architecture")
    cat = a.get("category") or ("Graph Algorithms" if is_dijkstra else "Topology & System")
    badge = a.get("badge") or ("DIJKSTRA ALGO" if is_dijkstra else "GRAPH TOPOLOGY")

    return {
        "graph_network": {
            "nodes": nodes,
            "edges": edges,
            "path": path,
            "breadcrumb": bc,
            "difficulty": diff,
            "category": cat,
            "badge": badge,
            "is_dijkstra": is_dijkstra,
        },
        "_illustrative": not bool(a.get("edges")),
    }


def _d_donut_chart(scene: Scene, lang: str) -> dict | None:
    a = _anim(scene)
    preset = a.get("preset")
    chart = _chart(scene)
    is_explicit = preset in ("donut_chart", "pie_chart", "donut", "pie") or (chart and chart.get("kind") in ("pie", "donut"))
    corpus = f"{scene.heading or ''} {scene.visual_prompt or ''} {scene.narration or ''} {' '.join(_bullets(scene))}".lower()
    has_kw = any(w in corpus for w in ("donut chart", "biểu đồ tròn", "tỷ lệ phần trăm", "phân bố", "market share", "thị phần"))

    if not is_explicit and not has_kw:
        return None

    labels = (chart.get("labels") if chart else None) or a.get("labels") or ["Transformer", "CNN", "RNN", "MLP"]
    values = (chart.get("values") if chart else None) or a.get("values") or [55, 25, 12, 8]
    unit = a.get("unit") or "%"
    bc = a.get("breadcrumb") or "BENCHMARK / ARCHITECTURE MARKET SHARE"
    diff = a.get("difficulty") or "Easy"
    cat = a.get("category") or "Statistics"

    return {
        "donut_chart": {
            "labels": labels,
            "values": values,
            "unit": unit,
            "breadcrumb": bc,
            "difficulty": diff,
            "category": cat,
        },
        "_illustrative": not bool(chart or a.get("values")),
    }


def _d_math_equation(scene: Scene, lang: str) -> dict | None:
    a = _anim(scene)
    preset = a.get("preset")
    is_explicit = preset in ("math_equation", "equation", "formula_breakdown", "formula")
    corpus = f"{scene.heading or ''} {scene.visual_prompt or ''} {scene.narration or ''} {' '.join(_bullets(scene))}".lower()
    has_kw = any(w in corpus for w in ("công thức toán", "phương trình", "bản chất công thức", "attention mechanism", "bayes", "gradient descent", "loss function", "định lý", "latex", "formula"))

    if not is_explicit and not has_kw:
        return None

    name = a.get("name") or a.get("title") or (scene.heading or "Transformer Attention")
    formula = (
        a.get("latex")
        or a.get("formula")
        or r"\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V"
    )
    parts = a.get("parts") or [
        {"symbol": r"QK^T", "name": "Độ tương đồng (Score)", "desc": "Tích vô hướng đo tương quan", "color": "#00f0ff"},
        {"symbol": r"\sqrt{d_k}", "name": "Scaling Factor", "desc": "Ổn định gradient khi chiều lớn", "color": "#ffe600"},
        {"symbol": r"\text{Softmax}", "name": "Chuẩn hoá xác suất", "desc": "Chuyển điểm số về tổng 1.0", "color": "#ff2bd6"},
        {"symbol": r"V", "name": "Value Matrix", "desc": "Tổng hợp ngữ nghĩa có trọng số", "color": "#39ff14"},
    ]
    bc = a.get("breadcrumb") or "MATHEMATICAL ESSENCE / EQUATION BREAKDOWN"
    diff = a.get("difficulty") or "Core Theory"
    cat = a.get("category") or "Mathematics"

    return {
        "math_equation": {
            "name": name,
            "formula": formula,
            "parts": parts,
            "breadcrumb": bc,
            "difficulty": diff,
            "category": cat,
        },
        "_illustrative": not bool(a.get("parts")),
    }


def _d_linear_transform(scene: Scene, lang: str) -> dict | None:
    a = _anim(scene)
    preset = a.get("preset")
    is_explicit = preset in ("linear_transform", "transform_2d", "matrix_grid")
    corpus = f"{scene.heading or ''} {scene.visual_prompt or ''} {scene.narration or ''} {' '.join(_bullets(scene))}".lower()
    has_kw = any(w in corpus for w in ("biến đổi tuyến tính", "linear transformation", "không gian vector", "vector cơ sở", "định thức", "determinant", "uốn cong lưới", "basis vectors"))

    if not is_explicit and not has_kw:
        return None

    matrix = a.get("matrix") or [[2, 1], [0.5, 1.5]]
    det = a.get("det") or (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0])
    bc = a.get("breadcrumb") or "3BLUE1BROWN / LINEAR ALGEBRA / MATRIX TRANSFORMATION"
    diff = a.get("difficulty") or "Medium"
    cat = a.get("category") or "Linear Algebra"

    return {
        "linear_transform": {
            "matrix": matrix,
            "det": round(float(det), 2),
            "ihat": [matrix[0][0], matrix[1][0]],
            "jhat": [matrix[0][1], matrix[1][1]],
            "breadcrumb": bc,
            "difficulty": diff,
            "category": cat,
        },
        "_illustrative": not bool(a.get("matrix")),
    }


def _d_calculus_curve(scene: Scene, lang: str) -> dict | None:
    a = _anim(scene)
    preset = a.get("preset")
    is_explicit = preset in ("calculus_curve", "derivative", "tangent_curve", "integral")
    corpus = f"{scene.heading or ''} {scene.visual_prompt or ''} {scene.narration or ''} {' '.join(_bullets(scene))}".lower()
    has_kw = any(w in corpus for w in ("đạo hàm", "tiếp tuyến", "tích phân", "calculus", "derivative", "tangent", "hệ số góc", "giải tích", "riemann"))

    has_func_data = bool(a.get("func_str") or a.get("func_name"))
    if not has_func_data and not has_kw:
        return None

    func_str = a.get("func_str") or a.get("func_name") or "f(x) = x³ - 2x + 1"
    x_val = a.get("x_val") or a.get("x0") or 1.2
    slope = a.get("slope") or a.get("tangent_slope") or 2.32
    bc = a.get("breadcrumb") or (f"{scene.heading.upper()} / CALCULUS" if scene.heading else "MATHEMATICAL VISUALIZATION / CALCULUS")
    diff = a.get("difficulty") or "Medium"
    cat = a.get("category") or "Calculus"

    return {
        "calculus_curve": {
            "func_str": func_str,
            "x_val": x_val,
            "slope": slope,
            "breadcrumb": bc,
            "difficulty": diff,
            "category": cat,
        },
        "_illustrative": not bool(a.get("slope")),
    }


def _pick_mascot(scene: Scene) -> str:
    """Tự động chọn linh vật: Bunny Girl (mặc định) hoặc Stickman (StykAnimate IK)."""
    corpus = f"{scene.heading or ''} {scene.visual_prompt or ''} {scene.narration or ''}".lower()
    vt = str(_g(scene, "visual_type", "")).lower()
    preset = str(_anim(scene).get("preset", "")).lower()
    if (
        any(w in corpus for w in ("stickman", "người que", "nguoi que", "stick", "pivot"))
        or preset in ("stickman", "stickman_explainer", "stick_figure")
        or vt in ("stickman", "challenge")
    ):
        return "stickman"
    return "bunny_girl"


def _d_story(scene: Scene, lang: str) -> dict | None:
    vt = str(_g(scene, "visual_type", "")).lower()
    items = _bullets(scene, 6) or _sentences(scene.narration, 4, 120)
    if not items:
        h = scene.heading or ""
        if h:
            items = [h]
    if not items and vt != "challenge":
        return None
    mascot = _pick_mascot(scene)
    if len(items) <= 1:
        narr_sents = _sentences(scene.narration, 4, 120)
        if len(narr_sents) >= 2:
            prob = [_truncate(x, 100) for x in narr_sents[:2]]
            sol = [_truncate(x, 100) for x in narr_sents[2:4]]
        else:
            prob = [items[0]] if items else ["Vấn đề kỹ thuật gặp phải"]
            sol = [_sentences(scene.narration, 1, 100)[0]] if _sentences(scene.narration, 1) else [_LABELS[lang]["solution"]]
        return {"story": {"problem": prob, "solution": sol, "mascot": mascot}}
    k = math.ceil(len(items) / 2)
    return {"story": {"problem": [_truncate(x, 100) for x in items[:k][:3]],
                      "solution": [_truncate(x, 100) for x in items[k:][:3]],
                      "mascot": mascot}}


def _d_mascot_tip(scene: Scene, lang: str) -> dict | None:
    vt = str(_g(scene, "visual_type", "")).lower()
    a = _anim(scene)
    preset = a.get("preset")
    is_explicit = vt in ("mascot_tip", "mascot", "tip", "pro_tip") or preset in ("mascot_tip", "mascot", "tip")
    corpus = f"{scene.heading or ''} {scene.visual_prompt or ''} {scene.narration or ''} {' '.join(_bullets(scene))}".lower()
    has_kw = any(w in corpus for w in ("pro-tip", "pro tip", "mẹo", "lời khuyên", "best practice", "chú ý", "lưu ý", "quy tắc vàng", "bí kíp", "kinh nghiệm", "tip hay"))
    if not is_explicit and not has_kw:
        return None
    bullets = _bullets(scene, 4) or _sentences(scene.narration, 3, 110)
    if not bullets:
        return None
    mascot = _pick_mascot(scene)
    mascot_names = {
        "bunny_girl": "Bunny Girl",
        "stickman": "Người Que StykAnimate",
    }
    title = scene.heading or f"Lời khuyên hữu ích từ {mascot_names.get(mascot, 'Bunny Girl')}"
    return {
        "mascot_tip": {
            "title": title,
            "tips": [_truncate(b, 110) for b in bullets[:3]],
            "mascot": mascot,
            "badge": "PRO-TIP & BEST PRACTICE",
            "pose": "explaining",
        }
    }


# ---------------------------------------------------------------------------
# Routing: chấm điểm cấu trúc + từ khoá, chọn template có dữ liệu
# ---------------------------------------------------------------------------

_KEYWORDS: dict[str, tuple[str, ...]] = {
    "conclusion": ("tổng kết", "tong ket", "kết luận", "ket luan", "takeaway", "takeaways", "ghi nhớ", "ghi nho",
                   "điểm cần nhớ", "diem can nho", "tóm lại", "tom lai", "key takeaways", "summary", "conclusion",
                   "đúc kết", "duc ket"),
    "quote_card": ("trích dẫn", "trich dan", "châm ngôn", "cham ngon", "bản chất cốt lõi", "ban chat cot loi",
                   "câu nói", "cau noi", "quote", "insight"),
    "terminal": ("terminal", "bash", "câu lệnh", "cau lenh", "pseudocode", "mã giả", "ma gia", "console", "command"),
    "matrix_op": ("ma trận", "ma tran", "matrix", "scalar", "vô hướng", "vo huong",
                  "nhân ma trận", "nhan ma tran", "đại số tuyến tính", "dai so tuyen tinh",
                  "linear algebra", "tensor", "vector", "tích vô hướng", "dot product"),
    "visual_algo": ("covariance", "hiệp phương sai", "hiep phuong sai", "thống kê", "thong ke",
                    "step by step", "từng bước", "tung buoc", "deep-ml", "worked example",
                    "bảng tính", "bang tinh", "công thức", "cong thuc", "formula"),
    "tree_graph": ("cây nhị phân", "cay nhi phan", "bst", "binary search tree", "binary tree",
                   "cây", "tree", "đồ thị", "graph", "dfs", "bfs", "traversal", "duyệt cây"),
    "two_pointers": ("hai con trỏ", "hai con tro", "two pointer", "two pointers",
                     "sliding window", "cửa sổ trượt", "cua so truot", "con trỏ", "con tro",
                     "trỏ trái", "trỏ phải", "left pointer", "right pointer"),
    "stack_queue": ("ngăn xếp", "ngan xep", "stack", "hàng đợi", "hang doi", "queue",
                    "lifo", "fifo", "push", "pop", "enqueue", "dequeue"),
    "dp_grid": ("quy hoạch động", "quy hoach dong", "dynamic programming", "dp table",
                "bảng dp", "bang dp", "lcs", "knapsack", "subproblem", "bài toán con"),
    "bar_benchmark": ("benchmark", "biểu đồ cột", "bieu do cot", "biểu đồ thanh", "bieu do thanh",
                      "so sánh hiệu năng", "so sanh hieu nang", "bar chart", "đo lường", "do luong"),
    "github_stats": ("github", "star", "tăng trưởng", "tang truong", "growth", "trend", "xu hướng", "xu huong"),
    "algo_trace": ("sorting", "sắp xếp", "sap xep", "bubble", "quicksort", "insertion", "selection", "mảng", "mang"),
    "neural_flow": ("neural", "mạng neural", "mang neural", "backprop", "forward pass", "activation"),
    "hud": ("counter", "bộ đếm", "bo dem", "throughput", "qps", "requests"),
    "scheduler": ("lịch", "lich", "phòng họp", "phong hop", "scheduling", "interval", "meeting",
                  "trùng lịch", "trung lich", "xếp lịch", "xep lich"),
    "cpu_scheduler": ("cpu scheduler", "lập lịch cpu", "lap lich cpu", "round robin", "tiến trình", "tien trinh",
                      "process scheduling", "scheduler"),
    "git_flow": ("git", "commit", "branch", "merge", "rebase", "pull request"),
    "attention_map": ("attention", "transformer", "self-attention", "query key value"),
    "diff_view": ("diff", "code review", "patch", "refactor"),
    "comparison": ("vs", "versus", "so sánh", "so sanh", "trade-off", "đánh đổi", "danh doi",
                   "khác biệt", "khac biet"),
    "metaphor": ("ẩn dụ", "an du", "đời thường", "doi thuong", "tưởng tượng", "tuong tuong",
                 "ví dụ đời sống", "vi du doi song", "mô hình đời thực", "metaphor", "analogy"),
    "stickman": ("nghịch lý", "nghich ly", "hỗn loạn", "hon loan", "bế tắc", "be tac",
                 "sai lầm", "sai lam", "bối rối", "boi roi", "paradox", "chaos", "mistake"),
    "flowchart": ("sơ đồ", "so do", "flowchart", "pipeline", "luồng xử lý", "luong xu ly",
                  "quy trình", "quy trinh", "data flow", "kiến trúc luồng", "process"),
    "system_layers": ("phân tầng", "phan tang", "system layers", "kiến trúc phân tầng",
                      "multi-tier", "clean architecture", "tầng kiến trúc", "layer", "tier", "layers"),
    "comparison_matrix": ("bảng so sánh", "bang so sanh", "ma trận", "ma tran", "bảng biểu", "bang bieu",
                          "tiêu chí", "tieu chi", "feature matrix", "so sánh chi tiết", "đặc tính kỹ thuật", "matrix table"),
    "radar_chart": ("radar", "biểu đồ radar", "bieu do radar", "mạng nhện", "mang nhen",
                    "spider chart", "trade-off", "đánh đổi đa chiều", "đa chiều"),
    "trie_tree": ("trie", "prefix tree", "cây tiền tố", "autocomplete", "chuỗi ký tự", "string matching"),
    "graph_network": ("dijkstra", "shortest path", "đường đi ngắn nhất", "đồ thị có trọng số",
                      "weighted graph", "network topology", "topo mạng", "mạng lưới"),
    "donut_chart": ("donut chart", "pie chart", "biểu đồ tròn", "tỷ lệ phần trăm", "phân bố",
                    "market share", "thị phần", "cơ cấu"),
    "math_equation": ("công thức toán", "phương trình", "bản chất công thức", "equation", "formula",
                      "attention mechanism", "bayes", "gradient descent", "loss function", "định lý"),
    "linear_transform": ("biến đổi tuyến tính", "linear transformation", "không gian vector", "vector cơ sở",
                         "định thức", "determinant", "uốn cong lưới", "basis vectors", "ma trận biến đổi"),
    "calculus_curve": ("đạo hàm", "tiếp tuyến", "tích phân", "calculus", "derivative", "tangent",
                      "hệ số góc", "giải tích", "riemann", "diện tích dưới đường cong"),
    "mascot_tip": ("pro-tip", "pro tip", "mẹo", "lời khuyên", "best practice", "chú ý", "lưu ý",
                   "quy tắc vàng", "bí kíp", "kinh nghiệm", "tip hay", "linh vật", "mascot"),
}
_PRIORITY = ["conclusion", "quote_card", "terminal", "matrix_op", "visual_algo", "tree_graph",
             "two_pointers", "stack_queue", "dp_grid", "math_equation", "linear_transform",
             "calculus_curve", "flowchart", "system_layers", "comparison_matrix", "radar_chart",
             "trie_tree", "graph_network", "donut_chart", "scheduler", "bar_benchmark", "github_stats",
             "algo_trace", "neural_flow", "hud", "diff_view", "cpu_scheduler", "git_flow",
             "attention_map", "comparison", "mascot_tip", "metaphor", "stickman", "architecture"]


def _kw_hits(text: str, kws: tuple[str, ...]) -> int:
    return sum(1 for k in kws if re.search(rf"(?<!\w){re.escape(k)}(?!\w)", text))


def _structural_scores(scene: Scene) -> dict[str, int]:
    st: dict[str, int] = {}
    vt = str(_g(scene, "visual_type", "")).lower()
    algo = str(_g(scene, "algorithm", "")).lower()
    preset = _anim(scene).get("preset")
    chart = _chart(scene)

    def add(k: str, v: int) -> None:
        st[k] = st.get(k, 0) + v

    if vt in ("conclusion", "summary", "takeaway", "takeaways", "outro"):
        add("conclusion", 20)
    elif vt in ("quote", "quotation"):
        add("quote_card", 20)
    elif vt in ("code", "terminal", "cmd"):
        add("terminal", 20)
    elif vt in ("matrix_op", "matrix"):
        add("matrix_op", 25)
    elif vt in ("visual_algo", "deep_ml"):
        add("visual_algo", 25)
    elif vt in ("tree_graph", "tree", "bst"):
        add("tree_graph", 25)
    elif vt in ("two_pointers", "sliding_window"):
        add("two_pointers", 25)
    elif vt in ("stack_queue", "stack", "queue"):
        add("stack_queue", 25)
    elif vt in ("dp_grid", "dp"):
        add("dp_grid", 25)
    elif vt in ("math_equation", "equation", "formula"):
        add("math_equation", 26)
    elif vt in ("linear_transform", "linear_algebra"):
        add("linear_transform", 26)
    elif vt in ("calculus_curve", "calculus", "derivative", "integral"):
        add("calculus_curve", 26)
    elif vt in ("trie_tree", "trie"):
        add("trie_tree", 26)
    elif vt in ("graph_network", "dijkstra"):
        add("graph_network", 26)
    elif vt in ("donut_chart", "pie_chart"):
        add("donut_chart", 26)
    elif vt in ("flowchart", "pipeline", "diagram"):
        add("flowchart", 26)
    elif vt in ("system_layers", "architecture_layers", "layers"):
        add("system_layers", 26)
    elif vt in ("comparison_matrix", "feature_matrix", "table", "benchmark_table"):
        add("comparison_matrix", 26)
    elif vt in ("radar_chart", "radar", "spider_chart"):
        add("radar_chart", 26)
    elif vt == "chart" and chart:
        if chart.get("kind") in ("pie", "donut"):
            add("donut_chart", 25)
        elif chart.get("kind") == "radar":
            add("radar_chart", 25)
        else:
            add("github_stats" if chart["kind"] == "line" else "bar_benchmark", 20)
    elif vt in ("git", "git_flow"):
        add("git_flow", 20)
    elif vt in ("diff", "diff_view"):
        add("diff_view", 20)
    elif vt in ("comparison", "compare"):
        add("comparison", 20)
    elif vt == "algorithm" and algo in ("matrix_mult", "linear_algebra"):
        add("matrix_op", 22)
    elif vt == "algorithm" and algo in ("covariance", "stats"):
        add("visual_algo", 22)
    elif vt == "algorithm" and algo in ("tree", "graph"):
        add("tree_graph", 22)
    elif vt == "algorithm" and algo in ("trie", "string"):
        add("trie_tree", 22)
    elif vt == "algorithm" and algo in ("dijkstra", "shortest_path"):
        add("graph_network", 22)
    elif vt == "algorithm" and algo in ("two_pointers", "linked_list"):
        add("two_pointers", 22)
    elif vt == "algorithm" and algo in ("stack_queue", "stack", "queue"):
        add("stack_queue", 22)
    elif vt == "algorithm" and algo in ("dp", "dynamic_programming"):
        add("dp_grid", 22)
    elif vt == "algorithm" and algo == "sorting":
        add("algo_trace", 20)
    elif vt == "algorithm" and algo == "neural_network":
        add("neural_flow", 20)
    elif vt in ("challenge", "stickman"):
        add("stickman", 30)
    elif vt in ("mascot_tip", "mascot", "tip", "pro_tip"):
        add("mascot_tip", 25)

    if preset in ("stickman", "stickman_explainer", "stick_figure"):
        add("stickman", 35)
    elif preset == "matrix_op":
        add("matrix_op", 25)
    elif preset == "visual_algo":
        add("visual_algo", 25)
    elif preset == "tree_graph":
        add("tree_graph", 25)
    elif preset == "two_pointers":
        add("two_pointers", 25)
    elif preset == "stack_queue":
        add("stack_queue", 25)
    elif preset == "dp_grid":
        add("dp_grid", 25)
    elif preset in ("math_equation", "equation"):
        add("math_equation", 25)
    elif preset in ("linear_transform", "transform_2d"):
        add("linear_transform", 25)
    elif preset in ("calculus_curve", "derivative", "integral"):
        add("calculus_curve", 25)
    elif preset in ("trie_tree", "trie"):
        add("trie_tree", 25)
    elif preset in ("graph_network", "dijkstra"):
        add("graph_network", 25)
    elif preset in ("donut_chart", "pie_chart"):
        add("donut_chart", 25)
    elif preset in ("flowchart", "pipeline", "flowchart_diagram"):
        add("flowchart", 25)
    elif preset in ("system_layers", "layers", "layer_stack"):
        add("system_layers", 25)
    elif preset in ("comparison_matrix", "feature_matrix", "table", "benchmark_table"):
        add("comparison_matrix", 25)
    elif preset in ("radar_chart", "radar", "spider_chart"):
        add("radar_chart", 25)
    elif preset in ("mascot_tip", "mascot", "tip"):
        add("mascot_tip", 25)
    elif preset == "bar_chart":
        add("bar_benchmark", 15)
    elif preset == "counter":
        add("hud", 18)
    elif preset == "sorting":
        add("algo_trace", 18)
    elif preset == "neural_net":
        add("neural_flow", 18)
    elif preset == "steps":
        add("architecture", 15)
    return st


def _plan_scene(scene: Scene, has_image: bool, lang: str) -> dict:
    """Trả {'template', 'data', 'illustrative'}; luôn có data['steps'] làm dự phòng."""
    heading = (scene.heading or "").lower()
    prompt = (scene.visual_prompt or "").lower()
    narr = (scene.narration or "")[:400].lower()
    bullets_text = " ".join(_bullets(scene)).lower()

    builders = {
        "conclusion": lambda: _d_takeaways(scene, lang),
        "quote_card": lambda: _d_quote(scene, lang),
        "terminal": lambda: _d_code(scene, lang),
        "matrix_op": lambda: _d_matrix_op(scene, lang),
        "visual_algo": lambda: _d_visual_algo(scene, lang),
        "tree_graph": lambda: _d_tree_graph(scene, lang),
        "two_pointers": lambda: _d_two_pointers(scene, lang),
        "stack_queue": lambda: _d_stack_queue(scene, lang),
        "dp_grid": lambda: _d_dp_grid(scene, lang),
        "math_equation": lambda: _d_math_equation(scene, lang),
        "linear_transform": lambda: _d_linear_transform(scene, lang),
        "calculus_curve": lambda: _d_calculus_curve(scene, lang),
        "trie_tree": lambda: _d_trie_tree(scene, lang),
        "graph_network": lambda: _d_graph_network(scene, lang),
        "donut_chart": lambda: _d_donut_chart(scene, lang),
        "flowchart": lambda: _d_flowchart(scene, lang),
        "system_layers": lambda: _d_system_layers(scene, lang),
        "comparison_matrix": lambda: _d_comparison_matrix(scene, lang),
        "radar_chart": lambda: _d_radar_chart(scene, lang),
        "bar_benchmark": lambda: _d_bars(scene, lang),
        "github_stats": lambda: _d_series(scene, lang),
        "algo_trace": lambda: _d_array(scene, lang),
        "neural_flow": lambda: _d_layers(scene, lang),
        "hud": lambda: _d_hud(scene, lang),
        "scheduler": lambda: _d_intervals(scene, lang),
        "cpu_scheduler": lambda: _d_tasks(scene, lang),
        "git_flow": lambda: _d_git(scene, lang),
        "attention_map": lambda: _d_attention(scene, lang),
        "diff_view": lambda: _d_diff(scene, lang),
        "comparison": lambda: _d_compare(scene, lang),
        "metaphor": lambda: _d_metaphor(scene, lang, has_image),
        "mascot_tip": lambda: _d_mascot_tip(scene, lang),
        "stickman": lambda: _d_story(scene, lang),
    }

    scores = _structural_scores(scene)
    for tpl, kws in _KEYWORDS.items():
        h = _kw_hits(heading, kws)
        p = _kw_hits(prompt, kws)
        b = _kw_hits(bullets_text, kws)
        n = _kw_hits(narr, kws)
        kw_score = (12 if tpl == "conclusion" else 4) * min(h, 2) + 3 * min(p, 2) + 2 * min(b, 2) + min(n, 2)
        if kw_score >= 2 or scores.get(tpl, 0) > 0:
            scores[tpl] = scores.get(tpl, 0) + kw_score
    # Tín hiệu từ chính dữ liệu: bullets dạng giờ họp / dòng diff +/- gần như chắc chắn
    if _d_intervals(scene, lang):
        scores["scheduler"] = scores.get("scheduler", 0) + 12
    if any(str(b).lstrip().startswith(("+", "-")) for b in (_g(scene, "bullets", []) or [])) and _d_diff(scene, lang):
        scores["diff_view"] = scores.get("diff_view", 0) + 24
    if has_image and "metaphor" in scores:
        scores["metaphor"] += 2

    ranked = sorted((t for t in scores if t in builders),
                    key=lambda t: (-scores[t], _PRIORITY.index(t) if t in _PRIORITY else 99))
    steps = _steps(scene, lang)
    for tpl in ranked:
        data = builders[tpl]()
        if data:
            illustrative = bool(data.pop("_illustrative", False))
            data["steps"] = steps
            return {"template": tpl, "data": data, "illustrative": illustrative}
    return {"template": "architecture", "data": {"steps": steps}, "illustrative": False}


def _detect_template(scene: Scene) -> str:
    """Tương thích ngược: chỉ trả tên template."""
    lang = "vi" if str(CONFIG.get("language", "vi")).lower().startswith("vi") else "en"
    return _plan_scene(scene, False, lang)["template"]


# ---------------------------------------------------------------------------
# HTML / JS
# ---------------------------------------------------------------------------

_HTML = r"""<!DOCTYPE html>
<html lang="__LANG__">
<head>
<meta charset="UTF-8">
<style>
__FONTFACE__
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { width: __WIDTH__px; height: __HEIGHT__px; background: #060a14; overflow: hidden; }
  #canvas { display: block; width: __WIDTH__px; height: __HEIGHT__px; }
</style>
<script>
window.MathJax = {
  svg: { fontCache: 'none' },
  startup: { typeset: false }
};
</script>
__MATHJAX_SCRIPT__
</head>
<body>
<canvas id="canvas" width="__WIDTH__" height="__HEIGHT__"></canvas>
<script>
const CFG = __CFG__;
</script>
<script>
__JS__
</script>
</body>
</html>
"""

_JS = r"""'use strict';
const W = CFG.width, H = CFG.height, FPS = CFG.fps, DURATION = Math.max(0.5, CFG.duration);
const TOTAL_FRAMES = Math.max(1, Math.round(DURATION * FPS));
const IS_VERTICAL = H > W;
const U = Math.min(W, H) / 1080;
const D = CFG.data || {};
const L = CFG.labels || {};
let TEMPLATE = CFG.template;
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const px = n => n * U;

let stockImg = null;
if (CFG.image) { stockImg = new Image(); stockImg.src = CFG.image; }

const MASCOT_IMGS = {};
if (CFG.mascot_images) {
  for (const [k, v] of Object.entries(CFG.mascot_images)) {
    const im = new Image();
    im.src = v;
    MASCOT_IMGS[k] = im;
  }
}

const CYAN = '#00f0ff', PINK = '#ff2bd6', GREEN = '#39ff14', YELLOW = '#ffe600', PURPLE = '#a855f7',
      ORANGE = '#ff6b35', BLUE = '#3b82f6', RED = '#ef4444', BG = '#060a14';
const BORDER = 'rgba(56,189,248,0.35)', CARD = 'rgba(12,20,40,0.92)', MUTED = '#94a3b8', DIM = '#64748b';
const PALETTE = [CYAN, PINK, GREEN, YELLOW, PURPLE, ORANGE, BLUE];
const FONT = "'MGFont','Be Vietnam Pro','Noto Sans','Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif";
const MONO = "'JetBrains Mono','Cascadia Mono','MGFont','Be Vietnam Pro',Consolas,'DejaVu Sans Mono','Noto Sans Mono',monospace";

// ---------- easing / math ----------
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const lerp = (a, b, t) => a + (b - a) * t;
const progress = (t, a, b) => clamp((t - a) / Math.max(b - a, 1e-6), 0, 1);
const easeOut = x => 1 - Math.pow(1 - x, 3);
const easeInOut = x => x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
const easeOutExpo = x => x >= 1 ? 1 : 1 - Math.pow(2, -10 * x);
const pop = (t, delay, dur = 0.5) => easeOutExpo(progress(t, delay, delay + dur));
function mulberry32(a) { return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a);
  t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function hashStr(s) { let h = 7; for (const c of String(s)) h = (h * 31 + c.charCodeAt(0)) | 0; return h; }
function fmtNum(n, maxDec = 2) {
  if (!isFinite(n)) return '0';
  const neg = n < 0; n = Math.abs(n);
  const s = Math.abs(n - Math.round(n)) < 1e-9 ? String(Math.round(n)) : n.toFixed(maxDec).replace(/\.?0+$/, '');
  let [i, d] = s.split('.');
  i = i.replace(/\B(?=(\d{3})+(?!\d))/g, CFG.lang === 'vi' ? '.' : ',');
  return (neg ? '-' : '') + i + (d ? (CFG.lang === 'vi' ? ',' : '.') + d : '');
}
function phaseOf(t, n) { return clamp((t - 0.6) / Math.max(DURATION - 1.2, 1), 0, 1) * n; }

// ---------- vẽ cơ bản ----------
function font(size, weight = 'normal', fam = FONT, style = '') {
  return `${style ? style + ' ' : ''}${weight} ${Math.max(8, Math.round(size))}px ${fam}`;
}
function roundRect(x, y, w, h, r) {
  const k = v => Math.max(0, Math.min(v || 0, Math.abs(w) / 2, Math.abs(h) / 2));
  const c = Array.isArray(r) ? r.map(k) : [k(r), k(r), k(r), k(r)];
  const [tl, tr, br, bl] = c.length === 4 ? c : [c[0], c[0], c[0], c[0]];
  ctx.beginPath();
  ctx.moveTo(x + tl, y);
  ctx.arcTo(x + w, y, x + w, y + h, tr);
  ctx.arcTo(x + w, y + h, x, y + h, br);
  ctx.arcTo(x, y + h, x, y, bl);
  ctx.arcTo(x, y, x + w, y, tl);
  ctx.closePath();
}
function panel(x, y, w, h, o = {}) {
  ctx.save();
  roundRect(x, y, w, h, o.rr || px(o.r === undefined ? 18 : o.r));
  ctx.fillStyle = o.fill || CARD; ctx.fill();
  if (o.stroke !== null) {
    ctx.strokeStyle = o.stroke || BORDER; ctx.lineWidth = px(o.lw || 2.5);
    if (o.glow) { ctx.shadowColor = o.gc || o.stroke || CYAN; ctx.shadowBlur = o.glow; }
    ctx.stroke();
  }
  ctx.restore();
}
function text(str, x, y, f, color, align = 'left', base = 'alphabetic') {
  ctx.font = f; ctx.fillStyle = color; ctx.textAlign = align; ctx.textBaseline = base; ctx.fillText(str, x, y);
}
function measure(str, f) { ctx.font = f; return ctx.measureText(str).width; }
function fitText(str, maxW, f) {
  ctx.font = f; str = String(str);
  if (ctx.measureText(str).width <= maxW) return str;
  while (str.length > 1 && ctx.measureText(str + '…').width > maxW) str = str.slice(0, -1);
  return str + '…';
}
function wrapLines(str, maxW, f, maxLines = 99) {
  ctx.save(); ctx.font = f;
  const words = String(str).split(/\s+/).filter(Boolean);
  const lines = []; let line = '';
  for (const w of words) {
    const test = line ? line + ' ' + w : w;
    if (ctx.measureText(test).width <= maxW) { line = test; continue; }
    if (line) lines.push(line);
    line = w;
    while (ctx.measureText(line).width > maxW && line.length > 1) {
      let k = line.length - 1;
      while (k > 1 && ctx.measureText(line.slice(0, k)).width > maxW) k--;
      lines.push(line.slice(0, k)); line = line.slice(k);
    }
  }
  if (line) lines.push(line);
  if (lines.length > maxLines) {
    lines.length = maxLines;
    let last = lines[maxLines - 1].replace(/[\s,.;:]+$/, '');
    while (last.length > 1 && ctx.measureText(last + '…').width > maxW) last = last.slice(0, -1);
    lines[maxLines - 1] = last + '…';
  }
  ctx.restore();
  return lines;
}
// Khối chữ nhiều dòng: y = mép trên. Trả về chiều cao.
function tb(str, x, y, maxW, fs, color, o = {}) {
  const f = font(fs, o.weight || 'normal', o.fam || FONT, o.style || '');
  const lh = fs * (o.lh || 1.3);
  const lines = wrapLines(str, maxW, f, o.maxLines || 99);
  lines.forEach((ln, i) => text(ln, x, y + (lh - fs) / 2 + fs * 0.88 + i * lh, f, color, o.align || 'left'));
  return lines.length * lh;
}
function tbHeight(str, maxW, fs, o = {}) {
  return wrapLines(str, maxW, font(fs, o.weight || 'normal', o.fam || FONT, o.style || ''), o.maxLines || 99).length * fs * (o.lh || 1.3);
}
function withScale(cx, cy, s, fn) { ctx.save(); ctx.translate(cx, cy); ctx.scale(s, s); ctx.translate(-cx, -cy); fn(); ctx.restore(); }
function glowStroke(color, blur, lw) { ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.shadowColor = color; ctx.shadowBlur = blur; }
function icon(name, cx, cy, s, color) {
  ctx.save(); ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = Math.max(2, s * 0.16);
  ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  if (name === 'check') { ctx.beginPath(); ctx.moveTo(cx - s * .5, cy); ctx.lineTo(cx - s * .15, cy + s * .35); ctx.lineTo(cx + s * .5, cy - s * .35); ctx.stroke(); }
  else if (name === 'cross') { ctx.beginPath(); ctx.moveTo(cx - s * .4, cy - s * .4); ctx.lineTo(cx + s * .4, cy + s * .4); ctx.moveTo(cx + s * .4, cy - s * .4); ctx.lineTo(cx - s * .4, cy + s * .4); ctx.stroke(); }
  else if (name === 'bolt') { ctx.beginPath(); ctx.moveTo(cx + s * .1, cy - s * .55); ctx.lineTo(cx - s * .35, cy + s * .1); ctx.lineTo(cx, cy + s * .1); ctx.lineTo(cx - s * .1, cy + s * .55); ctx.lineTo(cx + s * .35, cy - s * .1); ctx.lineTo(cx, cy - s * .1); ctx.closePath(); ctx.fill(); }
  else if (name === 'warn') { ctx.beginPath(); ctx.moveTo(cx, cy - s * .5); ctx.lineTo(cx + s * .55, cy + s * .45); ctx.lineTo(cx - s * .55, cy + s * .45); ctx.closePath(); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx, cy - s * .12); ctx.lineTo(cx, cy + s * .15); ctx.stroke(); ctx.beginPath(); ctx.arc(cx, cy + s * .3, s * .04, 0, 7); ctx.fill(); }
  else if (name === 'bulb') { ctx.beginPath(); ctx.arc(cx, cy - s * .1, s * .38, 0, Math.PI * 2); ctx.stroke(); ctx.beginPath(); ctx.moveTo(cx - s * .18, cy + s * .42); ctx.lineTo(cx + s * .18, cy + s * .42); ctx.stroke(); }
  else { ctx.beginPath(); ctx.arc(cx, cy, s * .25, 0, Math.PI * 2); ctx.fill(); }
  ctx.restore();
}

// ---------- render toán học KaTeX / MathJax (Option 2) ----------
const MATH_CACHE = new Map();
async function getMathImage(latex, color = '#ffffff') {
  if (!latex || !window.MathJax || !MathJax.tex2svg) return null;
  const raw = String(latex).trim();
  if (!raw) return null;
  const key = color + '::' + raw;
  if (MATH_CACHE.has(key)) return MATH_CACHE.get(key);
  try {
    const node = MathJax.tex2svg(raw, { display: true });
    const svg = node.querySelector('svg');
    if (!svg) return null;
    svg.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
    svg.setAttribute('color', color);
    svg.style.color = color;
    const blob = new Blob([svg.outerHTML], { type: 'image/svg+xml;charset=utf-8' });
    const img = new Image();
    const url = URL.createObjectURL(blob);
    await new Promise((res, rej) => {
      img.onload = () => { URL.revokeObjectURL(url); res(); };
      img.onerror = () => { URL.revokeObjectURL(url); rej(); };
      img.src = url;
    });
    MATH_CACHE.set(key, img);
    return img;
  } catch (e) {
    return null;
  }
}

function drawMath(latex, cx, cy, maxW, maxH, color = '#ffffff', opts = {}) {
  if (!latex) return false;
  const raw = String(latex).trim();
  const key = color + '::' + raw;
  const img = MATH_CACHE.get(key) || MATH_CACHE.get('#ffffff::' + raw);
  if (!img || !img.naturalWidth || !img.naturalHeight) return false;

  const natW = img.naturalWidth, natH = img.naturalHeight;
  let drawW, drawH;
  if (opts.targetH) {
    drawH = Math.min(opts.targetH, maxH);
    drawW = (drawH / natH) * natW;
    if (drawW > maxW) {
      drawW = maxW;
      drawH = (drawW / natW) * natH;
    }
  } else {
    const scale = Math.min(maxW / natW, maxH / natH, 3.0);
    drawW = natW * scale;
    drawH = natH * scale;
  }

  const x = opts.align === 'left' ? cx : (opts.align === 'right' ? cx - drawW : cx - drawW / 2);
  const y = cy - drawH / 2;

  ctx.save();
  if (opts.glow) {
    ctx.shadowColor = opts.glowColor || color;
    ctx.shadowBlur = opts.glow;
  }
  ctx.drawImage(img, x, y, drawW, drawH);
  ctx.restore();
  return true;
}

// ---------- nền / header / bố cục ----------
function drawBackground(t) {
  ctx.fillStyle = BG; ctx.fillRect(0, 0, W, H);
  const g = ctx.createRadialGradient(W * 0.3 + Math.sin(t * 0.5) * px(50), H * 0.4, px(80), W / 2, H / 2, W * 0.85);
  g.addColorStop(0, 'rgba(14,165,233,0.10)'); g.addColorStop(0.5, 'rgba(168,85,247,0.06)'); g.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
  const gs = px(IS_VERTICAL ? 50 : 65);
  ctx.fillStyle = 'rgba(56,189,248,0.07)'; ctx.beginPath();
  for (let x = gs / 2; x < W; x += gs) for (let y = gs / 2; y < H; y += gs) { ctx.moveTo(x + 1.2, y); ctx.arc(x, y, 1.2, 0, Math.PI * 2); }
  ctx.fill();
}
let HL = null;
function headerLayout() {
  if (HL) return HL;
  const meta = (D.matrix_op || D.visual_algo || D.tree_graph || D.two_pointers || D.stack_queue || D.dp_grid || D.flowchart || D.system_layers || D.comparison_matrix || D.radar_chart || D.math_equation || D.linear_transform || D.calculus_curve || D.trie_tree || D.graph_network || D.donut_chart || {});
  const hasMeta = !!(meta.breadcrumb || D.breadcrumb || meta.category || D.category);
  const mx = px(72);
  const top = px(IS_VERTICAL ? (hasMeta ? 130 : 96) : (hasMeta ? 78 : 48));
  const tSize = px(IS_VERTICAL ? 52 : 56);
  const maxW = W - mx * 2 - px(26);
  const tLines = wrapLines(CFG.title, maxW, font(tSize, 'bold'), 2);
  const lh = tSize * 1.2;
  const sSize = px(IS_VERTICAL ? 26 : 30);
  const sLines = CFG.subtitle ? wrapLines(CFG.subtitle, maxW, font(sSize), 2) : [];
  const sLh = sSize * 1.35;
  const bottom = top + tLines.length * lh + (sLines.length ? px(10) + sLines.length * sLh : 0);
  HL = { mx, top, tSize, tLines, lh, sSize, sLines, sLh, bottom, hasMeta, meta };
  return HL;
}
function drawHeader(t) {
  const h = headerLayout();
  const enter = easeOutExpo(clamp(t / 0.7, 0, 1));
  ctx.save(); ctx.globalAlpha = enter; ctx.translate(0, (1 - enter) * -px(24));

  // Breadcrumb bar & pills for Deep-ML style
  const meta = h.meta;
  const bc = meta.breadcrumb || D.breadcrumb;
  const diff = meta.difficulty || D.difficulty;
  const cat = meta.category || D.category;

  if (h.hasMeta) {
    const topBarY = h.top - px(22);
    if (bc) {
      text(String(bc), h.mx, topBarY, font(px(IS_VERTICAL ? 18 : 22), 'bold', FONT), '#94a3b8');
    }
    // Draw pills on right
    let pillRight = W - h.mx;
    if (cat) {
      const catText = String(cat);
      const catF = font(px(16), 'bold', FONT);
      ctx.font = catF;
      const catW = ctx.measureText(catText).width + px(24);
      pillRight -= catW;
      panel(pillRight, topBarY - px(19), catW, px(28), { r: 14, fill: 'rgba(168,85,247,0.18)', stroke: PURPLE, lw: 1.5 });
      text(catText, pillRight + catW / 2, topBarY - px(2), catF, PURPLE, 'center');
      pillRight -= px(10);
    }
    if (diff) {
      const diffText = '● ' + String(diff);
      const diffF = font(px(16), 'bold', FONT);
      ctx.font = diffF;
      const diffW = ctx.measureText(diffText).width + px(24);
      pillRight -= diffW;
      panel(pillRight, topBarY - px(19), diffW, px(28), { r: 14, fill: 'rgba(57,255,20,0.18)', stroke: GREEN, lw: 1.5 });
      text(diffText, pillRight + diffW / 2, topBarY - px(2), diffF, GREEN, 'center');
    }
  }

  ctx.save(); ctx.fillStyle = CYAN; ctx.shadowColor = CYAN; ctx.shadowBlur = px(16);
  ctx.fillRect(h.mx, h.top + px(4), px(8), h.tLines.length * h.lh - px(8)); ctx.restore();
  h.tLines.forEach((ln, i) => text(ln, h.mx + px(26), h.top + h.tSize * 0.95 + i * h.lh, font(h.tSize, 'bold'), '#f1f5f9'));
  const sy = h.top + h.tLines.length * h.lh + px(10);
  h.sLines.forEach((ln, i) => text(ln, h.mx + px(26), sy + h.sSize * 1.0 + i * h.sLh, font(h.sSize), MUTED));
  ctx.restore();
  return h.bottom;
}
function contentBox(bottom) {
  const mx = px(72), y = bottom + px(36), pad = px(IS_VERTICAL ? 230 : 70);
  return { x: mx, y, w: W - 2 * mx, h: Math.max(px(200), H - y - pad) };
}
function drawBadge(box) {
  const f = font(px(20), 'bold', FONT), tw = measure(L.illustrative, f) + px(32);
  panel(box.x + box.w - tw, box.y - px(34), tw, px(32), { r: 16, fill: 'rgba(255,230,0,0.10)', stroke: YELLOW, lw: 1.5 });
  text(L.illustrative, box.x + box.w - tw / 2, box.y - px(12), f, YELLOW, 'center');
}
function drawBrackets(x, y, w, h, col) {
  const arm = px(12), lw = px(3);
  ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(x + arm, y); ctx.lineTo(x, y); ctx.lineTo(x, y + h); ctx.lineTo(x + arm, y + h);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x + w - arm, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w - arm, y + h);
  ctx.stroke();
  ctx.restore();
}

// ---------- Người que StykAnimate (IK 2-Bone + Rig Chuẩn Pivot) ----------
function solveIK2Bone(bx, by, tx, ty, l1, l2, bendDir) {
  const dx = tx - bx, dy = ty - by, d = Math.hypot(dx, dy) || 1e-6;
  const tot = l1 + l2, eps = 1e-6;
  const ux = dx / d, uy = dy / d;
  if (d >= tot - eps) {
    return { mx: bx + ux * l1, my: by + uy * l1, ex: bx + ux * tot, ey: by + uy * tot };
  }
  const minD = Math.abs(l1 - l2);
  const cd = Math.max(d, minD + eps);
  const along = (l1 * l1 - l2 * l2 + cd * cd) / (2 * cd);
  const h = Math.sqrt(Math.max(0, l1 * l1 - along * along)) * (bendDir || 1);
  return {
    mx: bx + ux * along - uy * h,
    my: by + uy * along + ux * h,
    ex: bx + ux * cd,
    ey: by + uy * cd
  };
}

function drawStickman(x, groundY, sc, pose, t, color) {
  const p = String(pose || 'idle').toLowerCase();
  const hr = 20, tl = 65, ual = 32, fal = 30, thl = 38, shl = 36, sw = 14, hw = 10;
  const col = color || '#00f0ff';

  ctx.save();
  ctx.translate(x, groundY);
  ctx.scale(sc, sc);

  // 1. Bóng đổ ellipse dưới mặt đất
  ctx.save();
  ctx.fillStyle = 'rgba(0, 0, 0, 0.35)';
  ctx.beginPath();
  ctx.ellipse(0, 0, 36, 10, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  // 2. Tính toán các toạ độ khớp theo pose bằng IK
  let pelX = 0, pelY = -72, torsoTilt = 0, headTilt = 0;
  let lfX = -hw - 6, lfY = 0, rfX = hw + 6, rfY = 0;
  let lhX = -sw - 12, lhY = -40, rhX = sw + 12, rhY = -40;
  let propType = '', propPos = [0, 0], eyeOffset = [2, 0];
  let aura = 0, auraCol = col;

  if (p.includes('walk')) {
    const sp = 5.0, ph = t * sp;
    const stride = 24, lift = 14;
    pelY += Math.abs(Math.sin(ph)) * 5 - 4;
    torsoTilt = 4;
    lfX = -hw + Math.sin(ph) * stride;
    lfY = -Math.max(0, -Math.cos(ph)) * lift;
    rfX = hw + Math.sin(ph + Math.PI) * stride;
    rfY = -Math.max(0, -Math.cos(ph + Math.PI)) * lift;
    lhX = -sw - Math.sin(ph) * 20;
    lhY = pelY - tl + 10 + 40;
    rhX = sw - Math.sin(ph + Math.PI) * 20;
    rhY = pelY - tl + 10 + 40;
    eyeOffset = [4, 0];
  } else if (p.includes('run')) {
    const sp = 8.5, ph = t * sp;
    const stride = 36, lift = 20;
    pelY += Math.abs(Math.sin(ph)) * 8 - 8;
    torsoTilt = 12;
    lfX = -hw + Math.sin(ph) * stride;
    lfY = -Math.max(0, -Math.cos(ph)) * lift;
    rfX = hw + Math.sin(ph + Math.PI) * stride;
    rfY = -Math.max(0, -Math.cos(ph + Math.PI)) * lift;
    lhX = -sw - Math.sin(ph) * 28;
    lhY = pelY - tl + 15 - Math.cos(ph) * 12;
    rhX = sw - Math.sin(ph + Math.PI) * 28;
    rhY = pelY - tl + 15 - Math.cos(ph + Math.PI) * 12;
    eyeOffset = [6, 1];
  } else if (p.includes('confused') || p.includes('problem') || p.includes('doubt')) {
    const sway = Math.sin(t * 2.5) * 4;
    pelX += sway * 0.5;
    torsoTilt = sway;
    headTilt = 8 + Math.sin(t * 3.0) * 3;
    lhX = -sw - 18; lhY = pelY - tl + 45;
    const scratch = Math.sin(t * 8.0) * 3;
    rhX = headTilt + hr * 0.8;
    rhY = pelY - tl - hr * 1.3 + scratch;
    propType = 'question';
    propPos = [headTilt, pelY - tl - hr * 2.2];
    eyeOffset = [1, -3];
  } else if (p.includes('idea') || p.includes('solution') || p.includes('aha')) {
    const jump = -Math.abs(Math.sin(t * 6.0)) * 16;
    pelY += jump;
    lfY += jump * 0.5; rfY += jump * 0.5;
    lhX = -sw - 26; lhY = pelY - tl - 30;
    rhX = sw + 26; rhY = pelY - tl - 30;
    propType = 'bulb';
    propPos = [0, pelY - tl - hr * 2.5];
    aura = 1.0;
    auraCol = '#ffe600';
    eyeOffset = [0, -2];
  } else if (p.includes('code') || p.includes('laptop') || p.includes('typing')) {
    pelY = -48;
    torsoTilt = 8;
    lfX = 18; lfY = 0; rfX = 22; rfY = 0;
    const lX = 34, lY = pelY - 12;
    lhX = lX - 4; lhY = lY + Math.sin(t * 18.0) * 3;
    rhX = lX + 8; rhY = lY + Math.cos(t * 18.0) * 3;
    propType = 'laptop';
    propPos = [lX, lY];
    eyeOffset = [4, 3];
  } else if (p.includes('point') || p.includes('explain') || p.includes('present')) {
    const breath = Math.sin(t * 3.0) * 2;
    pelY += breath;
    lhX = -hw - 8; lhY = pelY - 4;
    rhX = sw + 55; rhY = pelY - tl - 15;
    propType = 'laser_pointer';
    propPos = [rhX, rhY];
    eyeOffset = [5, -1];
  } else if (p.includes('celebrate') || p.includes('win')) {
    const bob = Math.sin(t * 5.0) * 4;
    pelY += bob;
    rhX = sw + 16; rhY = pelY - tl - 36;
    lhX = -sw + 6; lhY = pelY - tl + 12;
    aura = 0.8;
    auraCol = '#39ff14';
    eyeOffset = [2, -4];
  } else {
    const breath = Math.sin(t * 3.0) * 3;
    pelY += breath;
    const armSw = Math.sin(t * 2.5) * 4;
    lhX = -sw - 12; lhY = pelY - tl + 50 + armSw;
    rhX = sw + 12; rhY = pelY - tl + 50 - armSw;
    eyeOffset = [2, 0];
  }

  const neckX = pelX + torsoTilt, neckY = pelY - tl;
  const headX = neckX + headTilt, headY = neckY - hr * 1.05;
  const lShX = neckX - sw, lShY = neckY + 4;
  const rShX = neckX + sw, rShY = neckY + 4;
  const lHipX = pelX - hw, lHipY = pelY;
  const rHipX = pelX + hw, rHipY = pelY;

  // Giải IK 2 chân
  const lLeg = solveIK2Bone(lHipX, lHipY, lfX, lfY, thl, shl, 1);
  const rLeg = solveIK2Bone(rHipX, rHipY, rfX, rfY, thl, shl, 1);

  // Giải IK 2 tay
  const lArm = solveIK2Bone(lShX, lShY, lhX, lhY, ual, fal, -1);
  const rArm = solveIK2Bone(rShX, rShY, rhX, rhY, ual, fal, 1);

  // Hào quang StykAnimate Aura
  if (aura > 0.05) {
    ctx.save();
    ctx.fillStyle = auraCol;
    ctx.shadowColor = auraCol;
    ctx.shadowBlur = 28 * aura;
    ctx.globalAlpha = 0.22 * aura;
    ctx.beginPath();
    ctx.arc(headX, headY, hr + 18, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  ctx.strokeStyle = col;
  ctx.fillStyle = col;
  ctx.lineWidth = 4;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.shadowColor = col;
  ctx.shadowBlur = 14;

  // Vẽ 2 chân
  // Chân trái
  ctx.beginPath();
  ctx.moveTo(lHipX, lHipY); ctx.lineTo(lLeg.mx, lLeg.my); ctx.lineTo(lLeg.ex, lLeg.ey);
  ctx.stroke();
  ctx.beginPath(); ctx.arc(lLeg.mx, lLeg.my, 2.5, 0, Math.PI * 2); ctx.fill();

  // Chân phải
  ctx.beginPath();
  ctx.moveTo(rHipX, rHipY); ctx.lineTo(rLeg.mx, rLeg.my); ctx.lineTo(rLeg.ex, rLeg.ey);
  ctx.stroke();
  ctx.beginPath(); ctx.arc(rLeg.mx, rLeg.my, 2.5, 0, Math.PI * 2); ctx.fill();

  // Khung thân & Vai, Hông
  ctx.beginPath(); ctx.moveTo(pelX, pelY); ctx.lineTo(neckX, neckY); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(lShX, lShY); ctx.lineTo(rShX, rShY); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(lHipX, lHipY); ctx.lineTo(rHipX, rHipY); ctx.stroke();

  // Vẽ 2 tay
  // Tay trái
  ctx.beginPath();
  ctx.moveTo(lShX, lShY); ctx.lineTo(lArm.mx, lArm.my); ctx.lineTo(lArm.ex, lArm.ey);
  ctx.stroke();
  ctx.beginPath(); ctx.arc(lArm.mx, lArm.my, 2.2, 0, Math.PI * 2); ctx.fill();

  // Tay phải
  ctx.beginPath();
  ctx.moveTo(rShX, rShY); ctx.lineTo(rArm.mx, rArm.my); ctx.lineTo(rArm.ex, rArm.ey);
  ctx.stroke();
  ctx.beginPath(); ctx.arc(rArm.mx, rArm.my, 2.2, 0, Math.PI * 2); ctx.fill();

  // Đầu với viền neon và nền tối
  ctx.save();
  ctx.fillStyle = '#0a0d14';
  ctx.beginPath();
  ctx.arc(headX, headY, hr, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  // Kính visor / Mắt phát sáng công nghệ
  ctx.fillStyle = YELLOW;
  ctx.shadowColor = YELLOW;
  ctx.shadowBlur = 10;
  ctx.beginPath();
  ctx.arc(headX + eyeOffset[0], headY + eyeOffset[1], 3.5, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  // Props (Vật phẩm đi kèm)
  if (propType === 'laser_pointer') {
    ctx.save();
    ctx.strokeStyle = '#ffe600';
    ctx.shadowColor = '#ffe600';
    ctx.shadowBlur = 16;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(rArm.ex, rArm.ey);
    ctx.lineTo(rArm.ex + 70, rArm.ey - 35);
    ctx.stroke();
    ctx.fillStyle = '#ffe600';
    ctx.beginPath();
    ctx.arc(rArm.ex + 70, rArm.ey - 35, 4.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  } else if (propType === 'bulb') {
    const bx = propPos[0], by = propPos[1];
    ctx.save();
    ctx.fillStyle = YELLOW;
    ctx.shadowColor = YELLOW;
    ctx.shadowBlur = 20;
    ctx.beginPath();
    ctx.arc(bx, by, 14, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = YELLOW;
    ctx.lineWidth = 2.5;
    for (let i = 0; i < 8; i++) {
      const a = i * Math.PI / 4 + t * 1.5;
      ctx.beginPath();
      ctx.moveTo(bx + Math.cos(a) * 18, by + Math.sin(a) * 18);
      ctx.lineTo(bx + Math.cos(a) * 28, by + Math.sin(a) * 28);
      ctx.stroke();
    }
    ctx.restore();
  } else if (propType === 'question') {
    const qx = propPos[0], qy = propPos[1];
    for (let q = 0; q < 3; q++) {
      const p = (t * 0.5 + q * 0.33) % 1, qa = Math.sin(p * Math.PI);
      ctx.save();
      ctx.fillStyle = `rgba(255, 230, 0, ${qa})`;
      ctx.font = font(26, 'bold');
      ctx.textAlign = 'center';
      ctx.shadowColor = YELLOW;
      ctx.shadowBlur = 14;
      ctx.fillText('?', qx + (q - 1) * 24 + Math.sin(t * 3 + q) * 10, qy - p * 40);
      ctx.restore();
    }
  } else if (propType === 'laptop') {
    const lx = propPos[0], ly = propPos[1];
    ctx.save();
    ctx.strokeStyle = CYAN;
    ctx.shadowColor = CYAN;
    ctx.shadowBlur = 12;
    ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(lx - 16, ly + 8); ctx.lineTo(lx + 20, ly + 8); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(lx + 18, ly + 8); ctx.lineTo(lx + 28, ly - 18); ctx.stroke();
    ctx.fillStyle = 'rgba(0, 240, 255, 0.15)';
    ctx.beginPath();
    ctx.moveTo(lx + 26, ly - 18); ctx.lineTo(lx + 18, ly + 8); ctx.lineTo(neckX + 8, neckY - 10);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  ctx.restore();
}

// Mascot Chính Kênh: Cô Bé Áo Khoác Tai Thỏ (Bunny Girl Chibi Asset)
function drawBunnyGirl(x, groundY, sc, pose, t, color) {
  const img = MASCOT_IMGS[pose] || MASCOT_IMGS.idle || MASCOT_IMGS.idea;
  if (!img || !img.complete || img.naturalWidth === 0) {
    return drawStickman(x, groundY, sc, pose, t, color);
  }

  // Physics Animation siêu mượt 60 FPS (Breathing, Squash & Stretch, Bobbing, Jumps)
  let bob = Math.sin(t * 4.5) * 4;
  let breathY = 1 + Math.sin(t * 3.5) * 0.025;
  let breathX = 1 - Math.sin(t * 3.5) * 0.015;
  let tilt = Math.sin(t * 2.2) * 0.025;
  let jump = 0;

  if (pose === 'idea') {
    jump = -Math.abs(Math.sin(t * 5.5)) * 14;
    tilt = -0.04 + Math.sin(t * 4) * 0.02;
    breathY = 1.03; breathX = 0.98;
  } else if (pose === 'confused') {
    tilt = 0.12 + Math.sin(t * 3) * 0.04;
    bob = Math.sin(t * 2.5) * 3;
  } else if (pose === 'happy') {
    jump = -Math.abs(Math.sin(t * 7)) * 20;
    tilt = Math.sin(t * 7) * 0.04;
    breathY = 1.05; breathX = 0.96;
  } else if (pose === 'explaining') {
    tilt = -0.04 + Math.sin(t * 2.2) * 0.025;
    bob = Math.sin(t * 3.5) * 3.5;
    breathY = 1 + Math.sin(t * 3.0) * 0.02;
    breathX = 1 - Math.sin(t * 3.0) * 0.015;
  }

  const targetH = 260 * sc;
  const oy = groundY - targetH - 10 + jump + bob;

  // 1. Bóng đổ dưới chân mượt mà theo độ cao nảy
  ctx.save();
  const shadowAlpha = Math.max(0.08, 0.24 + jump * 0.006);
  const shadowW = Math.max(20, (65 + jump * 0.3) * sc);
  ctx.fillStyle = `rgba(251, 191, 36, ${shadowAlpha})`;
  ctx.beginPath();
  ctx.ellipse(x, groundY, shadowW, 14 * sc, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  // 2. Vẽ hình ảnh nhân vật Sprite
  ctx.save();
  ctx.translate(x, oy + targetH);
  ctx.scale(sc * breathX, sc * breathY);
  ctx.rotate(tilt);

  const asp = (img.naturalWidth && img.naturalHeight) ? (img.naturalWidth / img.naturalHeight) : 1;
  const drawW = 260 * asp;
  const drawH = 260;

  // Viền hào quang phát sáng nhẹ
  ctx.save();
  ctx.shadowColor = (pose === 'idea' || pose === 'happy') ? '#fbbf24' : (pose === 'explaining' ? '#00f0ff' : '#38bdf8');
  ctx.shadowBlur = 18;
  ctx.drawImage(img, -drawW / 2, -drawH, drawW, drawH);
  ctx.restore();

  // 3. Hiệu ứng hạt động (Particle VFX)
  if (pose === 'confused') {
    for (let q = 0; q < 3; q++) {
      const p = (t * 0.5 + q * 0.33) % 1, qa = Math.sin(p * Math.PI);
      ctx.save();
      ctx.fillStyle = `rgba(255,230,0,${qa})`;
      ctx.font = font(28, 'bold');
      ctx.textAlign = 'center';
      ctx.shadowColor = YELLOW; ctx.shadowBlur = 14;
      ctx.fillText('?', (q - 1) * 26 + Math.sin(t * 3 + q) * 12 + 10, -drawH - 20 - p * 60);
      ctx.restore();
    }
  } else if (pose === 'idea') {
    const rays = 8, rl = 18 + Math.sin(t * 8) * 6;
    ctx.save();
    ctx.translate(35, -drawH - 30);
    ctx.strokeStyle = YELLOW; ctx.lineWidth = 2.5; ctx.shadowColor = YELLOW; ctx.shadowBlur = 18;
    for (let i = 0; i < rays; i++) {
      const a = i * Math.PI * 2 / rays + t * 0.6;
      ctx.beginPath();
      ctx.moveTo(Math.cos(a) * 20, Math.sin(a) * 20);
      ctx.lineTo(Math.cos(a) * (20 + rl), Math.sin(a) * (20 + rl));
      ctx.stroke();
    }
    ctx.restore();
  } else if (pose === 'happy') {
    for (let s = 0; s < 4; s++) {
      const p = (t * 0.6 + s * 0.25) % 1, sa = Math.sin(p * Math.PI);
      const sx = (s % 2 === 0 ? -1 : 1) * (50 + s * 15) + Math.sin(t * 4 + s) * 10;
      const sy = -drawH * 0.6 - p * 120;
      ctx.save();
      ctx.fillStyle = `rgba(251,191,36,${sa})`;
      ctx.shadowColor = YELLOW; ctx.shadowBlur = 12;
      ctx.beginPath();
      ctx.arc(sx, sy, 4 + Math.sin(t * 6 + s) * 2, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
  } else if (pose === 'explaining') {
    // Holographic tech pointer & presentation sparkles
    const ptx = drawW * 0.32, pty = -drawH * 0.54;
    const pulse = (t * 2.5) % 1;
    ctx.save();
    // 1. Vòng tròn holographic radar target
    ctx.strokeStyle = `rgba(0, 240, 255, ${1 - pulse})`;
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    ctx.arc(ptx, pty, 6 + pulse * 20, 0, Math.PI * 2);
    ctx.stroke();

    // 2. Điểm phát sáng laser tại đầu ngón tay / que chỉ
    ctx.fillStyle = '#00f0ff';
    ctx.shadowColor = '#00f0ff';
    ctx.shadowBlur = 12;
    ctx.beginPath();
    ctx.arc(ptx, pty, 4.5, 0, Math.PI * 2);
    ctx.fill();

    // 3. Tia laser năng lượng chỉ về phía bảng phân tích
    const bLen = 85 + Math.sin(t * 4) * 12;
    const bAngle = -0.20 + Math.sin(t * 2) * 0.04;
    ctx.strokeStyle = 'rgba(0, 240, 255, 0.7)';
    ctx.lineWidth = 2.2;
    ctx.setLineDash([7, 5]);
    ctx.lineDashOffset = -t * 28;
    ctx.beginPath();
    ctx.moveTo(ptx, pty);
    ctx.lineTo(ptx + Math.cos(bAngle) * bLen, pty + Math.sin(bAngle) * bLen);
    ctx.stroke();
    ctx.setLineDash([]);

    // 4. Hạt bụi tri thức holographic bay về phía nội dung
    for (let p = 0; p < 4; p++) {
      const prog = (t * 0.7 + p * 0.25) % 1;
      const pa = Math.sin(prog * Math.PI);
      const px_pos = ptx + 20 + prog * 90;
      const py_pos = pty - 12 - Math.sin(prog * Math.PI * 1.4) * 30 + (p % 2 === 0 ? -8 : 8);
      ctx.fillStyle = `rgba(56, 189, 248, ${pa * 0.9})`;
      ctx.shadowColor = '#38bdf8';
      ctx.shadowBlur = 8;
      ctx.beginPath();
      ctx.arc(px_pos, py_pos, 2.5 + Math.sin(t * 6 + p), 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }
  ctx.restore();
}

// Router điều hướng linh vật: Bunny Girl & Stickman
function drawMascot(type, x, groundY, sc, pose, t, color) {
  const m = String(type || 'bunny_girl').toLowerCase();
  if (m.includes('bunny') || m.includes('girl') || m.includes('tho') || m.includes('thỏ') || m.includes('anime')) {
    return drawBunnyGirl(x, groundY, sc, pose, t, color);
  }
  return drawStickman(x, groundY, sc, pose, t, color);
}

// ================= TEMPLATES (t, box) =================
const TEMPLATES = {};

// --- architecture: các bước tuần tự (fallback chung) ---
TEMPLATES.architecture = function (t, box) {
  const steps = (D.steps || []).slice(0, 5), n = steps.length; if (!n) return;
  const ph = phaseOf(t, n), active = Math.min(n - 1, Math.floor(ph));
  const gap = px(28), horiz = !IS_VERTICAL && n > 1;
  const fs = px(n <= 3 ? 32 : 28);
  let cw, ch, pos;
  if (horiz) {
    cw = Math.min(px(360), (box.w - gap * (n - 1)) / n); ch = Math.min(box.h * 0.8, px(320));
    const x0 = box.x + (box.w - (n * cw + gap * (n - 1))) / 2, y0 = box.y + (box.h - ch) / 2;
    pos = i => ({ x: x0 + i * (cw + gap), y: y0 });
  } else {
    cw = Math.min(box.w, px(900)); ch = Math.min(px(200), (box.h - gap * (n - 1)) / n);
    const y0 = box.y + (box.h - (n * ch + gap * (n - 1))) / 2, x0 = box.x + (box.w - cw) / 2;
    pos = i => ({ x: x0, y: y0 + i * (ch + gap) });
  }
  if (n > 1) { // đường dẫn + gói tin
    const a = pos(0), b = pos(n - 1);
    const ax = horiz ? a.x + cw / 2 : a.x + cw / 2, ay = horiz ? a.y + ch / 2 : a.y + ch / 2;
    const bx = horiz ? b.x + cw / 2 : b.x + cw / 2, by = horiz ? b.y + ch / 2 : b.y + ch / 2;
    ctx.save(); ctx.strokeStyle = 'rgba(0,240,255,0.30)'; ctx.lineWidth = px(4); ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke(); ctx.restore();
    const f = clamp((ph - 0.5) / (n - 1), 0, 1), px0 = lerp(ax, bx, f), py0 = lerp(ay, by, f);
    ctx.save(); ctx.fillStyle = GREEN; ctx.shadowColor = GREEN; ctx.shadowBlur = px(22); ctx.beginPath(); ctx.arc(px0, py0, px(10), 0, 7); ctx.fill(); ctx.restore();
  }
  steps.forEach((s, i) => {
    const p = pos(i), e = pop(t, i * 0.2, 0.5), on = i === active;
    withScale(p.x + cw / 2, p.y + ch / 2, e, () => {
      ctx.globalAlpha = e;
      panel(p.x, p.y, cw, ch, { stroke: on ? PINK : BORDER, glow: on ? px(22) : px(8), gc: on ? PINK : CYAN });
      text('0' + (i + 1), p.x + px(22), p.y + px(36), font(px(22), 'bold', MONO), on ? PINK : CYAN);
      const maxL = Math.max(2, Math.floor((ch - px(54)) / (fs * 1.3)));
      const th = tbHeight(s, cw - px(44), fs, { weight: 'bold', maxLines: maxL });
      tb(s, p.x + px(22), p.y + px(48) + Math.max(0, (ch - px(60) - th) / 2), cw - px(44), fs, '#f1f5f9', { weight: 'bold', maxLines: maxL });
    });
  });
};

// --- conclusion: thẻ ghi nhớ từ bullets ---
TEMPLATES.conclusion = function (t, box) {
  const items = (D.takeaways || []).slice(0, 4), n = items.length; if (!n) return;
  const gap = px(22), cw = Math.min(box.w, px(1300)), x0 = box.x + (box.w - cw) / 2;
  const ch = Math.min(px(190), (box.h - gap * (n - 1)) / n);
  const y0 = box.y + (box.h - (n * ch + gap * (n - 1))) / 2, step = Math.min(3.2, DURATION * 0.7 / n);
  const colors = [GREEN, ORANGE, CYAN, PINK];
  items.forEach((it, i) => {
    const e = pop(t, 0.8 + i * step, 0.6), col = colors[i % 4], y = y0 + i * (ch + gap);
    if (e <= 0) return;
    withScale(x0 + cw / 2, y + ch / 2, e, () => {
      ctx.globalAlpha = e;
      panel(x0, y, cw, ch, { fill: 'rgba(10,18,36,0.94)', stroke: col, glow: px(14) });
      const tag = Math.min(px(64), ch - px(32));
      panel(x0 + px(18), y + (ch - tag) / 2, tag, tag, { r: 12, fill: col, stroke: null });
      text(String(i + 1).padStart(2, '0'), x0 + px(18) + tag / 2, y + ch / 2 + px(9), font(px(26), 'bold', MONO), BG, 'center');
      const tx = x0 + px(18) + tag + px(24), tw = cw - (tx - x0) - px(24);
      const m = it.match(/^(.{2,40}?):\s+(.+)$/);
      if (m) {
        let bfs = px(26);
        while (bfs > px(18) && wrapLines(m[2], tw, font(bfs), 99).length > 2) bfs -= px(1.5);
        const hh = tbHeight(m[1], tw, px(28), { weight: 'bold', maxLines: 1 }), bh = tbHeight(m[2], tw, bfs, { maxLines: 3 });
        const top = y + (ch - hh - bh) / 2;
        tb(m[1], tx, top, tw, px(28), col, { weight: 'bold', maxLines: 1 });
        tb(m[2], tx, top + hh, tw, bfs, '#cbd5e1', { maxLines: 3 });
      } else {
        let fs = px(30);
        while (fs > px(20) && wrapLines(it, tw, font(fs, 'bold'), 99).length > 3) fs -= px(1.5);
        const hh = tbHeight(it, tw, fs, { weight: 'bold', maxLines: 3 });
        tb(it, tx, y + (ch - hh) / 2, tw, fs, '#f1f5f9', { weight: 'bold', maxLines: 3 });
      }
    });
  });
};

// --- quote_card ---
TEMPLATES.quote_card = function (t, box) {
  const q = D.quote || ''; if (!q) return;
  const cw = Math.min(box.w, px(1200)), ch = Math.min(box.h, px(600));
  const x = box.x + (box.w - cw) / 2, y = box.y + (box.h - ch) / 2, cx = x + cw / 2, cy = y + ch / 2;
  const e = pop(t, 0, 0.7);
  let fs = px(48);
  while (fs > px(26) && tbHeight(q, cw - px(200), fs, { weight: 'bold', style: 'italic' }) > ch - px(230)) fs -= px(2);
  withScale(cx, cy, e, () => {
    ctx.globalAlpha = e;
    panel(x, y, cw, ch, { r: 22, fill: 'rgba(18,12,38,0.95)', stroke: PURPLE, lw: 3, glow: px(28 + Math.sin(t * 3) * 8) });
    const tag = L.quote_tag, tf = font(px(20), 'bold'), tw = measure(tag, tf) + px(48);
    panel(cx - tw / 2, y - px(20), tw, px(40), { r: 20, fill: 'rgba(168,85,247,0.25)', stroke: PURPLE, lw: 1.5 });
    text(tag, cx, y + px(7), tf, '#e9d5ff', 'center');
    text('\u201C', x + px(52), y + px(150), font(px(150), 'bold', 'Georgia, serif'), 'rgba(168,85,247,0.35)');
    const th = tbHeight(q, cw - px(200), fs, { weight: 'bold', style: 'italic' });
    tb(q, cx, y + px(70) + (ch - px(100) - th) / 2, cw - px(200), fs, '#f8fafc', { weight: 'bold', style: 'italic', align: 'center' });
  });
};

// --- terminal ---
const KEYWORDS = new Set(['def', 'return', 'class', 'if', 'else', 'elif', 'for', 'while', 'in', 'not', 'and', 'or', 'import', 'from', 'break',
  'continue', 'lambda', 'None', 'True', 'False', 'FOR', 'WHILE', 'IF', 'ELSE', 'THEN', 'RETURN', 'FUNCTION', 'END', 'DO', 'mỗi', 'nếu', 'trả']);
function tokenize(line) {
  const out = [], re = /(#.*$|\/\/.*$)|("[^"]*"|'[^']*')|(\b\d+(?:\.\d+)?\b)|([A-Za-z_\u00C0-\u1EF9][\w\u00C0-\u1EF9]*)|(\s+|.)/gu;
  let m;
  while ((m = re.exec(line)) !== null) {
    if (m[1]) out.push([m[0], '#6b7a90']); else if (m[2]) out.push([m[0], YELLOW]); else if (m[3]) out.push([m[0], ORANGE]);
    else if (m[4]) out.push([m[0], KEYWORDS.has(m[0]) ? PINK : '#e2e8f0']); else out.push([m[0], MUTED]);
  }
  return out;
}
TEMPLATES.terminal = function (t, box) {
  const cd = D.code || { lines: [], lang: '' }; const lines0 = cd.lines || []; if (!lines0.length) return;
  const tw = Math.min(box.w, px(1500)), x = box.x + (box.w - tw) / 2;
  const headH = px(56), lh = Math.min(px(54), (box.h - headH - px(40)) / lines0.length), fs = lh * 0.55;
  const th = Math.min(box.h, headH + px(40) + lh * lines0.length), y = box.y + Math.max(0, (box.h - th) / 3);
  const e = pop(t, 0, 0.6), mf = font(fs, 'bold', MONO), gut = px(70);
  ctx.save(); ctx.globalAlpha = e;
  panel(x, y, tw, th, { r: 18, fill: 'rgba(6,10,22,0.97)', stroke: 'rgba(0,240,255,0.40)', glow: px(24) });
  roundRect(x, y, tw, headH, [px(18), px(18), 0, 0]); ctx.fillStyle = 'rgba(20,30,50,0.85)'; ctx.fill();
  ['#ff5f56', '#ffbd2e', '#27c93f'].forEach((c, i) => { ctx.fillStyle = c; ctx.beginPath(); ctx.arc(x + px(28 + i * 28), y + headH / 2, px(9), 0, 7); ctx.fill(); });
  text(cd.lang || 'pseudocode', x + tw / 2, y + headH / 2 + px(7), font(px(20), 'bold', MONO), DIM, 'center');
  ctx.font = mf; const cw1 = ctx.measureText('M').width, maxCh = Math.floor((tw - gut - px(40)) / cw1);
  const lines = lines0.map(l => l.length > maxCh ? l.slice(0, maxCh - 1) + '…' : l);
  const total = lines.reduce((a, l) => a + l.length + 1, 0);
  const cps = clamp(total / (DURATION * 0.6), 18, 90);
  let shown = Math.floor(Math.max(0, t - 0.5) * cps), lastY = y + headH + px(36), lastX = x + gut;
  for (let i = 0; i < lines.length; i++) {
    const ly = y + headH + px(36) + i * lh;
    if (shown <= 0 && i > 0) break;
    text(String(i + 1), x + px(24), ly, mf, '#475569');
    let cx = x + gut, left = shown;
    for (const [seg, col] of tokenize(lines[i])) {
      if (left <= 0) break;
      const s = seg.length > left ? seg.slice(0, left) : seg;
      text(s, cx, ly, mf, col); cx += measure(s, mf); left -= s.length;
    }
    shown -= lines[i].length + 1; lastY = ly; lastX = cx;
  }
  if (Math.floor(t * 2) % 2 === 0) { ctx.fillStyle = CYAN; ctx.fillRect(lastX + px(4), lastY - fs * 0.9, px(12), fs * 1.1); }
  ctx.restore();
};

// --- hai khung song song / chồng (dùng cho diff & comparison) ---
function twoPanels(box) {
  const gap = px(IS_VERTICAL ? 28 : 40);
  if (IS_VERTICAL) { const h = Math.min(px(560), (box.h - gap) / 2); const y0 = box.y + (box.h - 2 * h - gap) / 2;
    return [{ x: box.x, y: y0, w: box.w, h }, { x: box.x, y: y0 + h + gap, w: box.w, h }]; }
  const w = (box.w - gap) / 2, h = Math.min(box.h, px(640)), y0 = box.y + (box.h - h) / 2;
  return [{ x: box.x, y: y0, w, h }, { x: box.x + w + gap, y: y0, w, h }];
}
TEMPLATES.diff_view = function (t, box) {
  const df = D.diff || { old: [], new: [] }, ps = twoPanels(box), sets = [df.old, df.new], cols = [RED, GREEN];
  const titles = [L.old, L.new], maxN = Math.max(1, sets[0].length, sets[1].length);
  sets.forEach((lines, k) => {
    const p = ps[k], e = pop(t, k * 0.25, 0.5), col = cols[k], hh = px(54);
    const lh = Math.min(px(48), (p.h - hh - px(24)) / maxN), fs = lh * 0.56, f = font(fs, 'bold', MONO);
    ctx.save(); ctx.globalAlpha = e;
    panel(p.x, p.y, p.w, p.h, { fill: k ? 'rgba(16,50,25,0.9)' : 'rgba(50,10,10,0.9)', stroke: col });
    roundRect(p.x, p.y, p.w, hh, [px(18), px(18), 0, 0]); ctx.fillStyle = k ? 'rgba(57,255,20,0.15)' : 'rgba(239,68,68,0.15)'; ctx.fill();
    text(titles[k], p.x + p.w / 2, p.y + hh / 2 + px(7), font(px(20), 'bold', MONO), MUTED, 'center');
    lines.forEach((ln, i) => {
      const le = pop(t, 0.35 + k * 0.15 + i * 0.12, 0.35);
      if (le <= 0) return;
      ctx.save(); ctx.globalAlpha = e * le;
      const ly = p.y + hh + px(12) + i * lh;
      ctx.fillStyle = k ? 'rgba(57,255,20,0.08)' : 'rgba(239,68,68,0.08)'; ctx.fillRect(p.x + px(6), ly, p.w - px(12), lh - px(4));
      text(k ? '+' : '-', p.x + px(20), ly + lh * 0.68, f, col);
      text(fitText(ln, p.w - px(80), f), p.x + px(52), ly + lh * 0.68, f, '#e2e8f0');
      ctx.restore();
    });
    ctx.restore();
  });
};
TEMPLATES.comparison = function (t, box) {
  const c = D.cmp; if (!c) return;
  const ps = twoPanels(box), sides = [c.left, c.right], cols = [PINK, CYAN];
  sides.forEach((s, k) => {
    const p = ps[k], e = pop(t, k * 0.3, 0.6), col = cols[k];
    withScale(p.x + p.w / 2, p.y + p.h / 2, e, () => {
      ctx.globalAlpha = e;
      panel(p.x, p.y, p.w, p.h, { fill: 'rgba(12,20,40,0.94)', stroke: col, glow: px(18) });
      const ty = p.y + px(26), th = tb(s.title, p.x + px(28), ty, p.w - px(56), px(36), col, { weight: 'bold', maxLines: 2 });
      const avail = p.h - th - px(70), rows = s.rows.slice(0, 4), rowH = rows.length ? Math.min(px(96), avail / rows.length) : 0;
      rows.forEach((r, i) => {
        const re = pop(t, 0.4 + k * 0.15 + i * 0.2, 0.4); if (re <= 0) return;
        ctx.save(); ctx.globalAlpha = e * re;
        const ry = ty + th + px(22) + i * rowH; icon('dot', p.x + px(44), ry + rowH * 0.35, px(26), col);
        tb(r, p.x + px(70), ry, p.w - px(100), px(28), '#e2e8f0', { maxLines: 2 }); ctx.restore();
      });
    });
  });
  const cx = box.x + box.w / 2, cy = IS_VERTICAL ? (ps[0].y + ps[0].h + ps[1].y) / 2 : ps[0].y + ps[0].h / 2;
  const e = pop(t, 0.5, 0.5);
  withScale(cx, cy, e, () => { panel(cx - px(34), cy - px(34), px(68), px(68), { rr: px(34), fill: BG, stroke: YELLOW, glow: px(14) }); text('VS', cx, cy + px(9), font(px(26), 'bold', MONO), YELLOW, 'center'); });
};

// --- bar_benchmark: thanh ngang, số liệu thật ---
TEMPLATES.bar_benchmark = function (t, box) {
  const b = D.bars; if (!b) return;
  const n = b.values.length, maxV = Math.max(...b.values.map(Math.abs), 1e-9);
  const labelW = Math.min(px(340), box.w * 0.30), barX = box.x + labelW + px(20);
  const vW = Math.min(px(IS_VERTICAL ? 280 : 360), box.w * 0.30), barMax = box.w - labelW - px(20) - vW;
  const rowH = Math.min(px(100), (box.h - px(110)) / n), barH = rowH * 0.56, y0 = box.y + px(10);
  ctx.save(); ctx.strokeStyle = 'rgba(100,116,139,0.5)'; ctx.lineWidth = px(2); ctx.beginPath(); ctx.moveTo(barX, y0); ctx.lineTo(barX, y0 + n * rowH); ctx.stroke(); ctx.restore();
  b.values.forEach((v, i) => {
    const col = PALETTE[i % PALETTE.length], g = easeOutExpo(progress(t, 0.3 + i * 0.25, 1.3 + i * 0.25)), y = y0 + i * rowH + (rowH - barH) / 2;
    const w = Math.max(px(4), Math.abs(v) / maxV * barMax * g);
    const lh = tbHeight(b.labels[i], labelW, px(26), { weight: 'bold', maxLines: 2 });
    tb(b.labels[i], barX - px(20), y + (barH - lh) / 2, labelW, px(26), '#cbd5e1', { weight: 'bold', maxLines: 2, align: 'right' });
    ctx.save(); roundRect(barX, y, w, barH, [0, px(10), px(10), 0]);
    const gr = ctx.createLinearGradient(barX, 0, barX + w, 0); gr.addColorStop(0, col + '55'); gr.addColorStop(1, col);
    ctx.fillStyle = gr; ctx.shadowColor = col; ctx.shadowBlur = px(14); ctx.fill(); ctx.restore();
    const valStr = fmtNum(v * g) + (b.unit ? ' ' + b.unit : '');
    let valFs = px(26);
    while (valFs > px(16) && measure(valStr, font(valFs, 'bold', MONO)) > vW - px(20)) valFs -= px(2);
    text(fitText(valStr, vW - px(16), font(valFs, 'bold', MONO)), barX + w + px(14), y + barH / 2 + px(valFs * 0.35), font(valFs, 'bold', MONO), col);
  });
  const posVals = b.values.filter(v => v > 0);
  const mn = posVals.length ? Math.min(...posVals) : 0;
  const mx = Math.max(...b.values);
  const ratio = mn > 0 ? mx / mn : 0;
  const all = pop(t, 0.3 + n * 0.25 + 1.0, 0.6);
  if (n >= 2 && isFinite(ratio) && ratio >= 1.5 && all > 0) {
    const msg = L.ratio.replace('{x}', fmtNum(ratio, 1)), f = font(px(28), 'bold'), w = measure(msg, f) + px(60), cx = box.x + box.w / 2, y = y0 + n * rowH + px(24);
    withScale(cx, y + px(26), all, () => { panel(cx - w / 2, y, w, px(52), { rr: px(26), fill: 'rgba(57,255,20,0.12)', stroke: GREEN, glow: px(16) }); text(msg, cx, y + px(35), f, GREEN, 'center'); });
  }
};

// --- github_stats: đường tăng trưởng từ chart line ---
TEMPLATES.github_stats = function (t, box) {
  const s = D.series; if (!s) return;
  const n = s.values.length, mx = Math.max(...s.values, 1e-9) * 1.1, mn0 = Math.min(0, ...s.values);
  const x0 = box.x + px(110), cw = box.w - px(130), top = box.y + px(190), ch = Math.min(box.h - px(270), px(520)), base = top + ch;
  const len = easeInOut(progress(t, 0.4, 0.4 + Math.min(DURATION * 0.7, 8)));
  const pos = len * (n - 1), i0 = Math.min(n - 2, Math.floor(pos)), cur = lerp(s.values[i0], s.values[i0 + 1], pos - i0);
  const X = i => x0 + i / (n - 1) * cw, Y = v => base - (v - mn0) / (mx - mn0 || 1) * ch;
  ctx.save(); ctx.strokeStyle = 'rgba(100,116,139,0.5)'; ctx.lineWidth = px(2); ctx.beginPath(); ctx.moveTo(x0, top); ctx.lineTo(x0, base); ctx.lineTo(x0 + cw, base); ctx.stroke();
  [0, 0.5, 1].forEach(k => { const v = mn0 + (mx - mn0) * k, y = Y(v); ctx.strokeStyle = 'rgba(100,116,139,0.15)'; ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x0 + cw, y); ctx.stroke();
    text(fmtNum(v, 1), x0 - px(12), y + px(7), font(px(20), 'normal', MONO), DIM, 'right'); }); ctx.restore();
  const pts = []; for (let i = 0; i <= i0; i++) pts.push([X(i), Y(s.values[i])]); pts.push([X(pos), Y(cur)]);
  ctx.save(); const gr = ctx.createLinearGradient(0, top, 0, base); gr.addColorStop(0, 'rgba(255,230,0,0.28)'); gr.addColorStop(1, 'rgba(255,230,0,0)');
  ctx.beginPath(); ctx.moveTo(pts[0][0], base); pts.forEach(p => ctx.lineTo(p[0], p[1])); ctx.lineTo(pts[pts.length - 1][0], base); ctx.closePath(); ctx.fillStyle = gr; ctx.fill();
  ctx.beginPath(); pts.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])); glowStroke(YELLOW, px(14), px(4)); ctx.lineJoin = 'round'; ctx.stroke(); ctx.restore();
  for (let i = 0; i <= i0; i++) { icon('dot', X(i), Y(s.values[i]), px(20), YELLOW);
    if (n <= 7 || i === 0 || i === n - 1) text(fitText(s.labels[i], cw / Math.min(n, 7), font(px(20))), X(i), base + px(34), font(px(20)), MUTED, 'center'); }
  ctx.save(); ctx.shadowColor = YELLOW; ctx.shadowBlur = px(26);
  const curStr = fmtNum(cur) + (s.unit ? ' ' + s.unit : '');
  let curFs = px(IS_VERTICAL ? 72 : 96);
  while (curFs > px(36) && measure(curStr, font(curFs, 'bold', MONO)) > box.w - px(40)) curFs -= px(4);
  text(curStr, box.x + box.w / 2, box.y + px(100), font(curFs, 'bold', MONO), YELLOW, 'center'); ctx.restore();
  text(fitText(s.labels[Math.min(n - 1, Math.round(pos))], box.w - px(40), font(px(28), 'bold')), box.x + box.w / 2, box.y + px(150), font(px(28), 'bold'), MUTED, 'center');
};

// --- hud: bộ đếm lớn ---
TEMPLATES.hud = function (t, box) {
  const h = D.hud; if (!h) return;
  const cx = box.x + box.w / 2, cy = box.y + box.h * (h.metrics.length ? 0.38 : 0.45), r = px(IS_VERTICAL ? 230 : 210);
  ctx.save(); ctx.translate(cx, cy); ctx.rotate(t * 1.1); ctx.strokeStyle = 'rgba(0,240,255,0.15)'; ctx.lineWidth = px(2);
  ctx.beginPath(); ctx.arc(0, 0, r, 0, 7); ctx.stroke();
  [CYAN, PINK].forEach((c, i) => { glowStroke(c, px(12), px(5)); ctx.beginPath(); ctx.arc(0, 0, r + px(24), i * Math.PI, i * Math.PI + Math.PI * 0.65); ctx.stroke(); }); ctx.restore();
  const p = easeOut(progress(t, 0.3, 0.3 + Math.min(DURATION * 0.7, 6))), val = h.value * p;
  let fs = px(110); const full = fmtNum(h.value); while (fs > px(40) && measure(full, font(fs, 'bold', MONO)) > r * 1.7) fs -= px(4);
  ctx.save(); ctx.shadowColor = CYAN; ctx.shadowBlur = px(28); text(fmtNum(val), cx, cy + fs * 0.3, font(fs, 'bold', MONO), '#f8fafc', 'center'); ctx.restore();
  if (h.unit) text(h.unit.toUpperCase(), cx, cy + fs * 0.3 + px(52), font(px(30), 'bold'), MUTED, 'center');
  const m = h.metrics, mw = Math.min(px(300), (box.w - px(30) * (m.length - 1)) / Math.max(m.length, 1)), tot = m.length * mw + (m.length - 1) * px(30);
  m.forEach((it, i) => {
    const e = pop(t, 0.6 + i * 0.2, 0.5), x = cx - tot / 2 + i * (mw + px(30)), y = box.y + box.h * 0.74, col = [GREEN, YELLOW, CYAN][i % 3];
    withScale(x + mw / 2, y + px(50), e, () => { ctx.globalAlpha = e; panel(x, y, mw, px(100), { stroke: col, glow: px(10) });
      text(fitText(it.value, mw - px(24), font(px(32), 'bold', MONO)), x + mw / 2, y + px(48), font(px(32), 'bold', MONO), col, 'center');
      text(fitText(it.label, mw - px(24), font(px(22))), x + mw / 2, y + px(82), font(px(22)), DIM, 'center'); });
  });
};

// --- scheduler: interval scheduling (lịch + phân phòng tham lam) ---
function overlapBands(ivs) {
  const ev = []; ivs.forEach((a, i) => ivs.forEach((b, j) => { if (j > i) { const s = Math.max(a.start, b.start), e = Math.min(a.end, b.end); if (e > s) ev.push([s, e]); } }));
  ev.sort((a, b) => a[0] - b[0]); const out = [];
  ev.forEach(([s, e]) => { const l = out[out.length - 1]; if (l && s <= l[1]) l[1] = Math.max(l[1], e); else out.push([s, e]); });
  return out;
}
TEMPLATES.scheduler = function (t, box) {
  const ivs = D.intervals || []; if (!ivs.length) return;
  const n = ivs.length, rooms = D.rooms, tsw = DURATION > 5 ? clamp(DURATION * 0.5, 2.5, DURATION - 2.5) : DURATION * 0.5;
  const pr = easeInOut(progress(t, tsw, tsw + 0.9));
  const step = [15, 30, 60, 120, 240].find(s => (Math.max(...ivs.map(i => i.end)) - Math.min(...ivs.map(i => i.start))) / s <= 8) || 240;
  const t0 = Math.floor(Math.min(...ivs.map(i => i.start)) / step) * step, t1 = Math.ceil(Math.max(...ivs.map(i => i.end)) / step) * step;
  const ax0 = box.x + px(170), ax1 = box.x + box.w - px(20), X = m => ax0 + (m - t0) / (t1 - t0 || 1) * (ax1 - ax0);
  const axisY = box.y + px(36), top = axisY + px(24), areaH = box.h - px(130), rows = Math.max(n, rooms), rowH = Math.min(px(92), areaH / rows), barH = rowH * 0.78;
  const fmtT = m => String(Math.floor(m / 60) % 24).padStart(2, '0') + ':' + String(m % 60).padStart(2, '0');
  ctx.save(); ctx.strokeStyle = 'rgba(255,255,255,0.15)'; ctx.lineWidth = px(1.5); ctx.beginPath(); ctx.moveTo(ax0, axisY); ctx.lineTo(ax1, axisY); ctx.stroke();
  for (let m = t0; m <= t1; m += step) { ctx.strokeStyle = 'rgba(255,255,255,0.08)'; ctx.beginPath(); ctx.moveTo(X(m), axisY); ctx.lineTo(X(m), top + rows * rowH); ctx.stroke();
    text(fmtT(m), X(m), axisY - px(10), font(px(20), 'normal', MONO), DIM, 'center'); } ctx.restore();
  const bandA = (1 - pr) * pop(t, 0.8 + n * 0.15, 0.5);
  if (bandA > 0.01) overlapBands(ivs).forEach(([s, e]) => { ctx.save(); ctx.globalAlpha = bandA * (0.4 + 0.1 * Math.sin(t * 6)); ctx.fillStyle = RED; ctx.fillRect(X(s), top - px(6), X(e) - X(s), n * rowH + px(12)); ctx.restore();
    ctx.save(); ctx.globalAlpha = bandA; icon('warn', (X(s) + X(e)) / 2, top - px(22), px(24), YELLOW); ctx.restore(); });
  if (pr > 0) for (let r = 0; r < rooms; r++) { ctx.save(); ctx.globalAlpha = pr; text(`${L.room} ${r + 1}`, box.x + px(10), top + r * rowH + rowH / 2 + px(8), font(px(24), 'bold'), MUTED); ctx.restore(); }
  ivs.forEach((iv, i) => {
    const e = pop(t, 0.2 + i * 0.15, 0.5), y = lerp(top + i * rowH, top + iv.room * rowH, pr) + (rowH - barH) / 2, x = X(iv.start), w = X(iv.end) - X(iv.start);
    const col = pr > 0.5 ? GREEN : CYAN; ctx.save(); ctx.globalAlpha = e; ctx.translate(lerp(-px(30), 0, e), 0);
    panel(x, y, w, barH, { r: 10, fill: pr > 0.5 ? 'rgba(57,255,20,0.2)' : 'rgba(0,240,255,0.2)', stroke: col, glow: px(10) });
    const f = font(px(22), 'bold'); text(fitText(iv.label, w - px(20), f), x + px(12), y + barH * 0.46, f, '#f1f5f9');
    const tf = font(px(17), 'normal', MONO), ts = fmtT(iv.start) + '-' + fmtT(iv.end); if (measure(ts, tf) < w - px(20)) text(ts, x + px(12), y + barH * 0.82, tf, MUTED); ctx.restore();
  });
  const done = pop(t, tsw + 0.9, 0.6);
  if (done > 0) { const msg = L.rooms_needed.replace('{n}', rooms), f = font(px(30), 'bold'), w = measure(msg, f) + px(64), cx = box.x + box.w / 2, y = box.y + box.h - px(60);
    withScale(cx, y + px(26), done, () => { panel(cx - w / 2, y, w, px(52), { rr: px(26), fill: 'rgba(57,255,20,0.12)', stroke: GREEN, glow: px(16) }); text(msg, cx, y + px(36), f, GREEN, 'center'); }); }
};

// --- stickman: vấn đề -> giải pháp từ bullets/narration (với mascot cute) ---
TEMPLATES.stickman = function (t, box) {
  const s = D.story; if (!s) return;
  const tsw = DURATION > 5 ? clamp(DURATION * 0.55, 2.5, DURATION - 2.5) : DURATION * 0.55, resolved = t >= tsw && s.solution.length > 0;
  const col = resolved ? GREEN : CYAN, sc = (IS_VERTICAL ? 1.5 : 1.9) * U;
  let pxL, pyT, pw, ph, gy, fx;
  if (IS_VERTICAL) { fx = box.x + box.w / 2; gy = box.y + px(365); pxL = box.x; pyT = gy + px(35); pw = box.w; ph = box.h - (pyT - box.y); }
  else { fx = box.x + box.w * 0.13; gy = box.y + box.h * 0.8; pxL = box.x + box.w * 0.28; pyT = box.y; pw = box.w * 0.72; ph = box.h; }
  const mascotType = s.mascot || 'stickman';
  drawMascot(mascotType, fx, gy, sc, resolved ? 'idea' : 'confused', t, col);
  const items = resolved ? s.solution : s.problem, th = resolved ? GREEN : RED, title = resolved ? L.solution : L.problem;
  const flash = resolved ? 1 - progress(t, tsw, tsw + 0.5) : 0;
  panel(pxL, pyT, pw, ph, { r: 20, fill: resolved ? 'rgba(10,30,20,0.94)' : 'rgba(24,12,18,0.92)', stroke: th, lw: 3, glow: px(20) + flash * px(30) });
  const hh = px(62); roundRect(pxL, pyT, pw, hh, [px(20), px(20), 0, 0]); ctx.fillStyle = resolved ? 'rgba(16,50,28,0.95)' : 'rgba(50,14,20,0.95)'; ctx.fill();
  icon(resolved ? 'bulb' : 'warn', pxL + px(40), pyT + hh / 2, px(30), th); text(title, pxL + px(76), pyT + hh / 2 + px(9), font(px(28), 'bold', FONT), th);
  const gap = px(16), ih = Math.min(px(150), (ph - hh - px(36) - gap * 2) / Math.max(items.length, 1));
  items.forEach((it, i) => {
    const e = pop(t, (resolved ? tsw + 0.4 : 0.5) + i * 0.4, 0.5), y = pyT + hh + px(20) + i * (ih + gap); if (e <= 0) return;
    ctx.save(); ctx.globalAlpha = e; ctx.translate(lerp(px(30), 0, e), 0);
    panel(pxL + px(20), y, pw - px(40), ih, { r: 12, fill: resolved ? 'rgba(57,255,20,0.10)' : 'rgba(239,68,68,0.10)', stroke: th, lw: 1.5 });
    icon(resolved ? 'check' : 'cross', pxL + px(58), y + ih / 2, px(28), th);
    const fs = px(IS_VERTICAL ? 28 : 30), h2 = tbHeight(it, pw - px(150), fs, { weight: 'bold', maxLines: 3 });
    tb(it, pxL + px(92), y + (ih - h2) / 2, pw - px(150), fs, '#f8fafc', { weight: 'bold', maxLines: 3 }); ctx.restore();
  });
};

// --- mascot_tip: Linh vật chia sẻ mẹo công nghệ & Best Practice (Speech Bubble) ---
TEMPLATES.mascot_tip = function (t, box) {
  const mt = D.mascot_tip; if (!mt) return TEMPLATES.architecture(t, box);
  const mascotType = mt.mascot || 'bunny_girl';
  const tips = mt.tips || ['Chia nhỏ vấn đề thành các hàm đơn giản', 'Viết test case trước khi tối ưu', 'Đọc kỹ log lỗi thay vì đoán mò'];
  const nTips = tips.length;
  const cardH = px(IS_VERTICAL ? 120 : 96);
  const cardY = box.y + box.h - cardH;
  const mainH = box.h - cardH - px(24);

  const sc = (IS_VERTICAL ? 1.4 : 1.7) * U;
  let fx, gy, bubbleX, bubbleY, bubbleW, bubbleH;

  if (IS_VERTICAL) {
    fx = box.x + box.w / 2;
    gy = box.y + px(365);
    bubbleX = box.x;
    bubbleY = gy + px(30);
    bubbleW = box.w;
    bubbleH = mainH - (bubbleY - box.y);
  } else {
    fx = box.x + box.w * 0.16;
    gy = box.y + mainH * 0.78;
    bubbleX = box.x + box.w * 0.30;
    bubbleY = box.y + px(10);
    bubbleW = box.w * 0.70;
    bubbleH = mainH - px(20);
  }

  // 1. Vẽ Mascot tương tác động
  const mascotPose = mt.pose || 'explaining';
  drawMascot(mascotType, fx, gy, sc, mascotPose, t, YELLOW);

  const strokeCol = mascotPose === 'explaining' ? CYAN : YELLOW;
  const badgeBg = mascotPose === 'explaining' ? 'rgba(0, 240, 255, 0.16)' : 'rgba(255, 230, 0, 0.16)';

  // 2. Khung thoại Speech Bubble bo cong phát sáng
  panel(bubbleX, bubbleY, bubbleW, bubbleH, { r: 18, fill: 'rgba(10, 18, 36, 0.95)', stroke: strokeCol, lw: 2.5, glow: px(16) });

  // Huy hiệu tiêu đề
  const headH = px(56);
  roundRect(bubbleX, bubbleY, bubbleW, headH, [px(18), px(18), 0, 0]);
  ctx.fillStyle = badgeBg; ctx.fill();
  const bTitle = mt.badge || (mascotPose === 'explaining' ? 'GIẢI THÍCH CHI TIẾT & PHÂN TÍCH' : 'PRO-TIP & BEST PRACTICE');
  const btf = font(px(IS_VERTICAL ? 17 : 20), 'bold', FONT);
  text(bTitle, bubbleX + px(24), bubbleY + headH / 2 + px(7), btf, strokeCol);

  // Danh sách các Tips
  const tipGap = px(12);
  const availTipH = bubbleH - headH - px(30);
  const tipH = Math.min(px(100), (availTipH - (nTips - 1) * tipGap) / nTips);

  tips.forEach((tipText, i) => {
    const ty = bubbleY + headH + px(18) + i * (tipH + tipGap);
    const e = pop(t, 0.3 + i * 0.22, 0.45);
    if (e <= 0) return;
    ctx.save();
    ctx.globalAlpha = e;
    ctx.translate(lerp(px(24), 0, e), 0);
    panel(bubbleX + px(18), ty, bubbleW - px(36), tipH, { r: 10, fill: 'rgba(30, 41, 59, 0.85)', stroke: 'rgba(100, 116, 139, 0.45)', lw: 1.2 });
    icon('check', bubbleX + px(38), ty + tipH / 2, px(24), GREEN);
    const fs = px(IS_VERTICAL ? 20 : 25);
    const th = tbHeight(tipText, bubbleW - px(100), fs, { weight: 'bold', maxLines: 2 });
    tb(tipText, bubbleX + px(68), ty + (tipH - th) / 2, bubbleW - px(100), fs, '#f1f5f9', { weight: 'bold', maxLines: 2 });
    ctx.restore();
  });

  // 3. Bottom Status Card
  panel(box.x, cardY, box.w, cardH, { r: 16, fill: 'rgba(10, 18, 36, 0.95)', stroke: GREEN, lw: 2, glow: px(16) });
  const sBadge = 'KEY WISDOM';
  const sbf = font(px(IS_VERTICAL ? 16 : 19), 'bold', FONT);
  const sbw = measure(sBadge, sbf) + px(24);
  panel(box.x + px(20), cardY + px(IS_VERTICAL ? 14 : 24), sbw, px(32), { r: 16, fill: 'rgba(57, 255, 20, 0.15)', stroke: GREEN, lw: 1.5 });
  text(sBadge, box.x + px(20) + sbw / 2, cardY + px(IS_VERTICAL ? 35 : 45), sbf, GREEN, 'center');

  const bottomMsg = mt.title || 'Áp dụng vào thực tế để tối ưu hiệu năng và code sạch.';
  const bmf = font(px(IS_VERTICAL ? 20 : 25), 'bold', FONT);
  text(fitText(bottomMsg, box.w - sbw - px(60), bmf), box.x + px(35) + sbw, cardY + px(IS_VERTICAL ? 36 : 46), bmf, '#ffffff');
};

// --- cpu_scheduler: hàng đợi tác vụ -> CPU, chạy tuần tự theo thời lượng thật ---
TEMPLATES.cpu_scheduler = function (t, box) {
  const tasks = D.tasks || [], n = tasks.length; if (!n) return;
  const known = tasks.every(k => k.dur != null && k.dur > 0), wsum = known ? tasks.reduce((a, k) => a + k.dur, 0) : n;
  const s0 = 0.8, s1 = Math.max(s0 + 1, DURATION - 0.8); let acc = 0; const slots = tasks.map(k => { const w = known ? k.dur : 1, a = s0 + acc / (wsum || 1) * (s1 - s0); acc += w; return { s: a, e: s0 + acc / (wsum || 1) * (s1 - s0) }; });
  const cur = slots.findIndex(sl => t >= sl.s && t < sl.e), doneN = slots.filter(sl => t >= sl.e).length;
  const perRow = n <= 3 ? n : Math.ceil(n / 2), rowsN = Math.ceil(n / perRow), gap = px(18), cwd = Math.min(px(300), (box.w - gap * (perRow - 1)) / perRow), chh = px(100);
  const qx0 = box.x + (box.w - (perRow * cwd + gap * (perRow - 1))) / 2, qy0 = box.y + px(36);
  text(L.queue, qx0, qy0 - px(10), font(px(20), 'bold'), '#38bdf8');
  const centers = [];
  tasks.forEach((k, i) => {
    const r = Math.floor(i / perRow), c = i % perRow, x = qx0 + c * (cwd + gap), y = qy0 + r * (chh + gap), e = pop(t, i * 0.12, 0.4);
    const run = i === cur, done = t >= slots[i].e; centers.push([x + cwd / 2, y + chh]); ctx.save(); ctx.globalAlpha = e * (done ? 0.55 : 1);
    panel(x, y, cwd, chh, { r: 12, stroke: run ? ORANGE : done ? GREEN : BORDER, glow: run ? px(18) : 0, lw: run ? 3 : 2 });
    text(fitText(k.name, cwd - px(24), font(px(24), 'bold')), x + px(14), y + px(40), font(px(24), 'bold'), run ? ORANGE : '#f1f5f9');
    text(fitText(k.dur != null ? fmtNum(k.dur) + ' ' + (k.unit || '') : `#${i + 1}`, cwd - px(24), font(px(20), 'normal', MONO)), x + px(14), y + px(76), font(px(20), 'normal', MONO), MUTED);
    if (done) icon('check', x + cwd - px(28), y + px(30), px(22), GREEN);
    if (run) { ctx.fillStyle = ORANGE; ctx.fillRect(x + px(10), y + chh - px(10), (cwd - px(20)) * progress(t, slots[i].s, slots[i].e), px(5)); } ctx.restore();
  });
  const size = px(IS_VERTICAL ? 260 : 230), cx = box.x + box.w / 2, cy = qy0 + rowsN * (chh + gap) + px(60), cpuY = Math.min(cy, box.y + box.h - size - px(70));
  if (cur >= 0) { ctx.save(); glowStroke(ORANGE, px(10), px(3)); ctx.setLineDash([px(10), px(8)]); ctx.lineDashOffset = -t * px(40); ctx.beginPath(); ctx.moveTo(centers[cur][0], centers[cur][1]); ctx.lineTo(cx, cpuY); ctx.stroke(); ctx.restore(); }
  const x = cx - size / 2; ctx.save(); ctx.strokeStyle = '#64748b'; ctx.lineWidth = px(3);
  for (let i = 0; i < 8; i++) { const f = (i + 0.5) / 8, a = x + f * size, b = cpuY + f * size;
    ctx.beginPath(); ctx.moveTo(a, cpuY - px(14)); ctx.lineTo(a, cpuY); ctx.moveTo(a, cpuY + size); ctx.lineTo(a, cpuY + size + px(14)); ctx.moveTo(x - px(14), b); ctx.lineTo(x, b); ctx.moveTo(x + size, b); ctx.lineTo(x + size + px(14), b); ctx.stroke(); } ctx.restore();
  panel(x, cpuY, size, size, { fill: 'rgba(15,23,42,0.96)', stroke: ORANGE, lw: 3, glow: px(24 + Math.sin(t * 5) * 8) });
  text('CPU', cx, cpuY + size * 0.34, font(px(34), 'bold', MONO), ORANGE, 'center');
  const label = cur >= 0 ? `${L.running}: ${tasks[cur].name}` : (doneN >= n ? L.done : '');
  text(fitText(label, size - px(24), font(px(20))), cx, cpuY + size * 0.56, font(px(20)), MUTED, 'center');
  text(`${doneN}/${n}`, cx, cpuY + size * 0.8, font(px(38), 'bold', MONO), GREEN, 'center');
  if (known) text(fitText(`${L.total}: ${fmtNum(wsum)} ${tasks[0].unit || ''}`, box.w - px(40), font(px(22), 'normal', MONO)), cx, cpuY + size + px(52), font(px(22), 'normal', MONO), DIM, 'center');
};

// --- git_flow ---
TEMPLATES.git_flow = function (t, box) {
  const cm = D.commits || []; const n = cm.length; if (n < 2) return;
  const lw = Math.min(px(180), (box.w - px(160)) / (n - 1) - px(12));
  const x0 = box.x + Math.max(px(70), lw / 2 + px(10)), x1 = box.x + box.w - Math.max(px(70), lw / 2 + px(10)), lineY = box.y + box.h * 0.6, brY = lineY - Math.min(px(200), box.h * 0.36);
  const X = i => lerp(x0, x1, i / (n - 1)), lp = easeInOut(progress(t, 0.2, 1.2));
  ctx.save(); glowStroke(GREEN, px(14), px(4)); ctx.beginPath(); ctx.moveTo(x0, lineY); ctx.lineTo(x0 + lp * (x1 - x0), lineY); ctx.stroke(); ctx.restore();
  const bp = easeInOut(progress(t, 0.9, 2.0)), bx0 = X(0), bx1 = X(n - 1), k = Math.min(px(60), (bx1 - bx0) / 5);
  ctx.save(); glowStroke(CYAN, px(10), px(3.5)); ctx.setLineDash([px(10), px(6)]); ctx.beginPath(); ctx.moveTo(bx0, lineY);
  const seg = [[bx0 + k * 0.6, lineY, bx0 + k * 0.6, brY, bx0 + k, brY], [bx1 - k, brY, bx1 - k * 0.6, brY, bx1 - k * 0.6, lineY]];
  if (bp > 0) { ctx.bezierCurveTo(...seg[0]); const mid = bx0 + k + (bx1 - k - bx0 - k) * clamp(bp * 1.4 - 0.2, 0, 1); ctx.lineTo(mid, brY); if (bp > 0.85) { ctx.lineTo(bx1 - k, brY); ctx.bezierCurveTo(seg[1][2], seg[1][3], seg[1][2], lineY, bx1, lineY); } }
  ctx.stroke(); ctx.restore();
  [0.33, 0.66].forEach((f, i) => { const bxm = lerp(bx0 + k, bx1 - k, f), e = pop(t, 1.2 + i * 0.3, 0.4); if (e > 0) { ctx.save(); ctx.globalAlpha = e; ctx.fillStyle = CYAN; ctx.shadowColor = CYAN; ctx.shadowBlur = px(14); ctx.beginPath(); ctx.arc(bxm, brY, px(11), 0, 7); ctx.fill(); ctx.restore(); } });
  const bl = pop(t, 1.0, 0.5); if (bl > 0) { ctx.save(); ctx.globalAlpha = bl; const f = font(px(24), 'bold', MONO), w = measure(L.branch, f) + px(40); panel(bx0 + k, brY - px(64), w, px(40), { rr: px(10), fill: 'rgba(0,240,255,0.12)', stroke: CYAN, lw: 2 }); text(L.branch, bx0 + k + w / 2, brY - px(36), f, CYAN, 'center'); ctx.restore(); }
  cm.forEach((c, i) => {
    const e = pop(t, 0.2 + i / (n - 1) * 1.0, 0.4); if (e <= 0) return;
    ctx.save(); ctx.globalAlpha = e; ctx.fillStyle = GREEN; ctx.shadowColor = GREEN; ctx.shadowBlur = px(18); ctx.beginPath(); ctx.arc(X(i), lineY, px(14), 0, 7); ctx.fill(); ctx.shadowBlur = 0;
    text(String(i + 1), X(i), lineY + px(6), font(px(16), 'bold', MONO), BG, 'center');
    const lw = Math.min(px(200), (x1 - x0) / (n - 1) - px(12)); tb(c, X(i), lineY + px(34), lw, px(24), '#cbd5e1', { align: 'center', maxLines: 3 }); ctx.restore();
  });
  const hp = easeInOut(progress(t, 0.8, Math.max(DURATION - 0.8, 1.8))), hx = lerp(X(0), X(n - 1), hp);
  ctx.save(); ctx.globalAlpha = clamp(t - 0.3, 0, 1); ctx.shadowColor = YELLOW; ctx.shadowBlur = px(22); text('HEAD', hx, lineY - px(34), font(px(24), 'bold', MONO), YELLOW, 'center');
  ctx.fillStyle = YELLOW; ctx.beginPath(); ctx.moveTo(hx, lineY - px(22)); ctx.lineTo(hx - px(8), lineY - px(32)); ctx.lineTo(hx + px(8), lineY - px(32)); ctx.closePath(); ctx.fill(); ctx.restore();
  const mf = font(px(24), 'bold', MONO); text('main', x0, lineY + px(IS_VERTICAL ? 150 : 140), mf, GREEN);
};

// --- attention_map (trọng số mô phỏng, softmax theo hàng) ---
TEMPLATES.attention_map = function (t, box) {
  const tok = D.tokens || [], n = tok.length; if (n < 2) return;
  const rng = mulberry32(hashStr(tok.join('|'))), Wm = [];
  for (let r = 0; r < n; r++) { const sc = []; for (let c = 0; c < n; c++) sc.push(-Math.abs(r - c) * 1.1 + (rng() - 0.5) * 1.6 + (r === c ? 1.2 : 0)); const m = Math.max(...sc), ex = sc.map(v => Math.exp(v - m)), z = ex.reduce((a, b) => a + b, 0); Wm.push(ex.map(v => v / z)); }
  const lab = px(IS_VERTICAL ? 130 : 170), cell = Math.min(px(120), (box.w - lab) / n, (box.h - px(150)) / n), gw = cell * n;
  const gx = box.x + (box.w - gw + lab) / 2, gy = box.y + px(100), reveal = progress(t, 0.2, 1.4);
  const mx = Math.max(...Wm.flat());
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) { const idx = r * n + c, ct = clamp(progress(reveal, idx / (n * n) * 0.6, idx / (n * n) * 0.6 + 0.3), 0, 1), w = Wm[r][c] / mx;
    ctx.fillStyle = `rgba(${Math.round(lerp(5, 0, w))},${Math.round(lerp(10, 240, w))},${Math.round(lerp(40, 255, w * 0.4))},${ct * 0.9})`; ctx.fillRect(gx + c * cell + 1, gy + r * cell + 1, cell - 2, cell - 2);
    if (cell >= px(64) && ct > 0.5) text(Wm[r][c].toFixed(2), gx + c * cell + cell / 2, gy + r * cell + cell / 2 + px(7), font(px(20), 'bold', MONO), `rgba(255,255,255,${ct * (0.4 + w * 0.6)})`, 'center'); }
  const lf = font(px(22), 'bold', MONO);
  tok.forEach((tk, i) => { const s = fitText(tk, cell * 1.0, lf); text(s, gx + i * cell + cell / 2, gy - px(14), lf, MUTED, 'center'); text(fitText(tk, lab - px(20), lf), gx - px(14), gy + i * cell + cell / 2 + px(8), lf, MUTED, 'right'); });
  text(`${L.key} \u2192`, gx + gw / 2, gy - px(52), font(px(22), 'bold'), CYAN, 'center');
  ctx.save(); ctx.translate(gx - lab + px(14), gy + gw / 2); ctx.rotate(-Math.PI / 2); text(`${L.query} \u2192`, 0, 0, font(px(22), 'bold'), CYAN, 'center'); ctx.restore();
  const fr = Math.floor(t / 1.6) % n; ctx.save(); glowStroke(CYAN, px(16), px(3)); ctx.setLineDash([px(6), px(4)]); ctx.strokeRect(gx, gy + fr * cell, gw, cell); ctx.restore();
};

// --- neural_flow ---
TEMPLATES.neural_flow = function (t, box) {
  const layers = D.layers || [], nL = layers.length; if (nL < 2) return;
  const x0 = box.x + px(50), x1 = box.x + box.w - px(50), y0 = box.y + px(30), y1 = box.y + box.h - px(110), stepX = (x1 - x0) / (nL - 1);
  const maxN = Math.max(...layers), nr = Math.min(px(26), (y1 - y0) / (maxN * 2.4)), wave = (t * 0.5) % (1.4) * (nL - 1) - 0.2;
  const NY = (li, ni) => y0 + (ni + 0.5) * (y1 - y0) / layers[li];
  for (let li = 0; li < nL - 1; li++) for (let a = 0; a < layers[li]; a++) for (let b = 0; b < layers[li + 1]; b++) {
    const hot = clamp(1 - Math.abs(wave - li - 0.5) / 0.8, 0, 1); ctx.strokeStyle = `rgba(56,189,248,${0.07 + 0.05 * Math.abs(Math.sin(a * 3 + b)) + hot * 0.25})`; ctx.lineWidth = px(1.2);
    ctx.beginPath(); ctx.moveTo(x0 + li * stepX, NY(li, a)); ctx.lineTo(x0 + (li + 1) * stepX, NY(li + 1, b)); ctx.stroke(); }
  for (let li = 0; li < nL; li++) {
    const lx = x0 + li * stepX, col = PALETTE[li % PALETTE.length], en = pop(t, li * 0.15, 0.4), act = clamp(1 - Math.abs(wave - li) / 0.7, 0, 1);
    for (let ni = 0; ni < layers[li]; ni++) { const ny = NY(li, ni), r = nr * en; ctx.save(); ctx.shadowColor = col; ctx.shadowBlur = px(8) + act * px(22);
      ctx.fillStyle = `rgba(6,10,20,0.9)`; ctx.beginPath(); ctx.arc(lx, ny, r, 0, 7); ctx.fill(); ctx.globalAlpha = en; ctx.strokeStyle = col; ctx.lineWidth = px(2.5); ctx.stroke();
      ctx.fillStyle = col; ctx.globalAlpha = en * (0.15 + act * 0.7); ctx.beginPath(); ctx.arc(lx, ny, r * 0.7, 0, 7); ctx.fill(); ctx.restore(); }
    const lbl = li === 0 ? L.input : li === nL - 1 ? L.output : `${L.hidden} ${li}`;
    ctx.save(); ctx.globalAlpha = pop(t, li * 0.15 + 0.2, 0.4); text(fitText(lbl, stepX - px(10), font(px(24), 'bold', MONO)), lx, y1 + px(44), font(px(24), 'bold', MONO), col, 'center'); text(`${layers[li]}`, lx, y1 + px(78), font(px(20), 'normal', MONO), DIM, 'center'); ctx.restore();
  }
};

// --- algo_trace: mô phỏng sắp xếp thật ---
function simulateSort(kind, input) {
  const a = input.slice(), steps = []; let cmp = 0, swp = 0; const sorted = new Set();
  const push = (hi, sw) => steps.push({ a: a.slice(), hi, sw, sorted: new Set(sorted), cmp, swp });
  push([], false);
  const swap = (i, j) => { [a[i], a[j]] = [a[j], a[i]]; swp++; };
  if (kind === 'insertion') {
    sorted.add(0);
    for (let i = 1; i < a.length; i++) { let j = i; while (j > 0) { cmp++; if (a[j - 1] > a[j]) { swap(j - 1, j); push([j - 1, j], true); j--; } else { push([j - 1, j], false); break; } } for (let k = 0; k <= i; k++) sorted.add(k); push([], false); }
  } else if (kind === 'selection') {
    for (let i = 0; i < a.length; i++) { let m = i; for (let j = i + 1; j < a.length; j++) { cmp++; push([m, j], false); if (a[j] < a[m]) m = j; } if (m !== i) { swap(i, m); push([i, m], true); } sorted.add(i); push([], false); }
  } else if (kind === 'quick') {
    const qs = (lo, hi) => { if (lo >= hi) { if (lo === hi) sorted.add(lo); return; } const p = a[hi]; let i = lo;
      for (let j = lo; j < hi; j++) { cmp++; push([j, hi], false); if (a[j] < p) { if (i !== j) { swap(i, j); push([i, j], true); } i++; } }
      if (i !== hi) { swap(i, hi); push([i, hi], true); } sorted.add(i); push([], false); qs(lo, i - 1); qs(i + 1, hi); };
    qs(0, a.length - 1);
  } else {
    for (let i = 0; i < a.length; i++) { for (let j = 0; j < a.length - i - 1; j++) { cmp++; if (a[j] > a[j + 1]) { swap(j, j + 1); push([j, j + 1], true); } else push([j, j + 1], false); } sorted.add(a.length - 1 - i); push([], false); }
  }
  for (let k = 0; k < a.length; k++) sorted.add(k); push([], false);
  return steps;
}
const SORT_NAMES = { bubble: 'Bubble Sort', insertion: 'Insertion Sort', selection: 'Selection Sort', quick: 'Quick Sort (Lomuto)' };
let SORT_STEPS = null;
TEMPLATES.algo_trace = function (t, box) {
  const arr = D.array || []; if (arr.length < 2) return;
  if (!SORT_STEPS) SORT_STEPS = simulateSort(D.algo || 'bubble', arr);
  const total = SORT_STEPS.length, avail = Math.max(DURATION - 1.6, 1), tps = clamp(avail / total, 0.1, 0.8);
  const idx = clamp(Math.floor((t - 0.8) / tps), 0, total - 1), st = SORT_STEPS[idx], n = arr.length;
  const maxV = Math.max(...arr.map(Math.abs), 1), minV = Math.min(0, ...arr), range = maxV - minV || 1;
  const baseY = box.y + box.h - px(60), maxH = Math.max(px(120), box.h - px(230)), bw = Math.min(px(120), box.w / n * 0.78), gap = Math.min(px(20), bw * 0.25), totalW = n * bw + (n - 1) * gap, sx = box.x + (box.w - totalW) / 2;
  text(fitText(SORT_NAMES[D.algo || 'bubble'], box.w, font(px(34), 'bold', MONO)), box.x, box.y + px(34), font(px(34), 'bold', MONO), PINK);
  const infoStr = `${L.step} ${idx} / ${total - 1}   \u00B7   ${L.compared}: ${st.cmp}   \u00B7   ${L.swaps}: ${st.swp}`;
  const infoFs = px(IS_VERTICAL ? 20 : 26);
  text(fitText(infoStr, box.w, font(infoFs, 'bold', MONO)), box.x, box.y + px(80), font(infoFs, 'bold', MONO), YELLOW);
  st.a.forEach((v, i) => {
    const bh = Math.max(px(8), (v - minV) / range * maxH), x = sx + i * (bw + gap), y = baseY - bh, hi = st.hi.includes(i), sw = hi && st.sw, ok = st.sorted.has(i);
    const col = sw ? PINK : hi ? YELLOW : ok ? GREEN : CYAN;
    ctx.save(); ctx.shadowColor = col; ctx.shadowBlur = hi ? px(24) : px(6); const gr = ctx.createLinearGradient(0, y, 0, baseY); gr.addColorStop(0, col); gr.addColorStop(1, col + '44'); ctx.fillStyle = gr; roundRect(x, y, bw, bh, [px(8), px(8), 0, 0]); ctx.fill(); ctx.restore();
    text(String(v), x + bw / 2, y - px(10), font(px(Math.min(30, bw / U * 0.34)), 'bold', MONO), '#e2e8f0', 'center');
    text(String(i), x + bw / 2, baseY + px(34), font(px(20), 'normal', MONO), DIM, 'center');
  });
};

// --- metaphor: ảnh minh hoạ + chú thích (hoặc ngăn kéo vẽ tay) ---
TEMPLATES.metaphor = function (t, box) {
  const caps = D.captions || [], cw = Math.min(box.w, px(1100)), ch = Math.min(box.h, px(IS_VERTICAL ? 1000 : 640)), x = box.x + (box.w - cw) / 2, y = box.y + (box.h - ch) / 2;
  const e = pop(t, 0, 0.6), capH = caps.length ? px(86) * Math.min(caps.length, 3) / (IS_VERTICAL ? 1 : 1.6) : 0, imgY = y + px(16), imgH = ch - px(32) - (IS_VERTICAL ? capH : px(90));
  withScale(x + cw / 2, y + ch / 2, e, () => {
    ctx.globalAlpha = e; panel(x, y, cw, ch, { r: 20, fill: 'rgba(10,18,36,0.95)', stroke: YELLOW, glow: px(20) });
    const ix = x + px(16), iw = cw - px(32);
    if (stockImg && stockImg.complete && stockImg.naturalWidth > 0) {
      ctx.save(); roundRect(ix, imgY, iw, imgH, px(14)); ctx.clip();
      const zoom = 1 + 0.08 * progress(t, 0, DURATION), s = Math.max(iw / stockImg.naturalWidth, imgH / stockImg.naturalHeight) * zoom, dw = stockImg.naturalWidth * s, dh = stockImg.naturalHeight * s;
      ctx.drawImage(stockImg, ix + (iw - dw) / 2, imgY + (imgH - dh) / 2, dw, dh); ctx.restore();
    } else {
      const n = 3, dh = (imgH - px(10) * (n - 1)) / n;
      for (let i = 0; i < n; i++) { const slide = i === 0 ? Math.sin(t * 2) * px(14) + px(14) : 0, yy = imgY + i * (dh + px(10)), hl = i === 0;
        panel(ix + slide, yy, iw - slide, dh, { r: 10, fill: hl ? 'rgba(255,230,0,0.16)' : 'rgba(30,41,59,0.8)', stroke: hl ? YELLOW : 'rgba(100,116,139,0.4)', lw: hl ? 2 : 1 });
        ctx.fillStyle = hl ? YELLOW : '#64748b'; roundRect(ix + slide + (iw - slide) / 2 - px(30), yy + dh / 2 - px(4), px(60), px(8), px(4)); ctx.fill();
        if (caps[i]) text(fitText(caps[i], iw - px(60), font(px(24), 'bold')), ix + slide + px(24), yy + px(34), font(px(24), 'bold'), hl ? YELLOW : '#cbd5e1'); }
    }
    if (caps.length && (stockImg && stockImg.complete && stockImg.naturalWidth > 0)) caps.slice(0, 3).forEach((c, i) => {
      const ce = pop(t, 0.8 + i * 0.4, 0.5); ctx.save(); ctx.globalAlpha = e * ce;
      if (IS_VERTICAL) { const yy = imgY + imgH + px(14) + i * px(80); panel(ix, yy, iw, px(68), { r: 12, stroke: PALETTE[i], lw: 1.5 }); text(fitText(c, iw - px(40), font(px(26), 'bold')), ix + px(20), yy + px(44), font(px(26), 'bold'), '#f1f5f9'); }
      else { const cwid = (iw - px(16) * (Math.min(caps.length, 3) - 1)) / Math.min(caps.length, 3), xx = ix + i * (cwid + px(16)), yy = imgY + imgH + px(14); panel(xx, yy, cwid, px(64), { r: 12, stroke: PALETTE[i], lw: 1.5 }); text(fitText(c, cwid - px(28), font(px(24), 'bold')), xx + px(14), yy + px(41), font(px(24), 'bold'), '#f1f5f9'); }
      ctx.restore(); });
  });
};

// --- matrix_op: đại số tuyến tính / phép nhân ma trận từng ô kiểu Deep-ML ---
TEMPLATES.matrix_op = function (t, box) {
  const m = D.matrix_op; if (!m) return;
  const rows = m.rows || 2, cols = m.cols || 3;
  const steps = m.steps || [];
  const nSteps = steps.length || 1;
  const availT = Math.max(DURATION - 1.2, 1.0);
  const tps = availT / nSteps;
  const curStepIdx = clamp(Math.floor((t - 0.6) / tps), 0, nSteps - 1);
  const curStep = steps[curStepIdx] || { row: 0, col: 0, expr: '', val: 0 };
  const stepProg = clamp((t - 0.6 - curStepIdx * tps) / tps, 0, 1);

  const calcH = px(IS_VERTICAL ? 120 : 96);
  const calcY = box.y + box.h - calcH;
  const matAreaH = box.h - calcH - px(24);

  if (!IS_VERTICAL) {
    const gap = px(12);
    const maxCellW = px(110);
    const maxCellH = px(84);
    const cellW = Math.min(maxCellW, (box.w * 0.36 - (cols - 1) * gap) / cols);
    const cellH = Math.min(maxCellH, (matAreaH - (rows - 1) * gap - px(40)) / rows);

    const matW = cols * cellW + (cols - 1) * gap;
    const matH = rows * cellH + (rows - 1) * gap;
    const scBoxW = Math.max(cellW, px(90));
    const scBoxH = Math.max(cellH, px(90));

    const totalOpW = scBoxW + px(50) + matW + px(50) + matW;
    const startX = box.x + Math.max(0, (box.w - totalOpW) / 2);
    const midY = box.y + matAreaH / 2;

    // 1. Scalar Box
    const scX = startX;
    const scY = midY - scBoxH / 2;
    panel(scX, scY, scBoxW, scBoxH, { r: 16, fill: 'rgba(245, 158, 11, 0.12)', stroke: '#f59e0b', lw: 2.5, glow: px(18) });
    text('Scalar c', scX + scBoxW / 2, scY - px(10), font(px(16), 'bold', MONO), '#f59e0b', 'center');
    text(String(m.scalar), scX + scBoxW / 2, scY + scBoxH / 2 + px(12), font(px(38), 'bold', MONO), '#ffffff', 'center');

    // Multiply Symbol
    text('×', scX + scBoxW + px(25), midY + px(10), font(px(34), 'bold', MONO), '#f59e0b', 'center');

    // 2. Matrix A
    const matAX = scX + scBoxW + px(50);
    const matAY = midY - matH / 2;
    text('Matrix A', matAX + matW / 2, matAY - px(10), font(px(18), 'bold', MONO), CYAN, 'center');
    drawBrackets(matAX - px(12), matAY, matW + px(24), matH, CYAN);

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const cx = matAX + c * (cellW + gap);
        const cy = matAY + r * (cellH + gap);
        const isActive = (r === curStep.row && c === curStep.col);
        const val = m.matrix_a[r] ? m.matrix_a[r][c] : 0;

        if (isActive) {
          panel(cx, cy, cellW, cellH, { r: 12, fill: 'rgba(255, 230, 0, 0.20)', stroke: YELLOW, lw: 3, glow: px(24) });
          text(String(val), cx + cellW / 2, cy + cellH / 2 + px(10), font(px(30), 'bold', MONO), YELLOW, 'center');
        } else {
          panel(cx, cy, cellW, cellH, { r: 12, fill: 'rgba(15, 23, 42, 0.90)', stroke: 'rgba(100, 116, 139, 0.4)', lw: 1.5 });
          text(String(val), cx + cellW / 2, cy + cellH / 2 + px(9), font(px(26), 'bold', MONO), '#cbd5e1', 'center');
        }
      }
    }

    // Equals Symbol
    text('=', matAX + matW + px(25), midY + px(10), font(px(34), 'bold', MONO), '#94a3b8', 'center');

    // 3. Result Matrix C
    const matCX = matAX + matW + px(50);
    const matCY = matAY;
    text('Result C', matCX + matW / 2, matCY - px(10), font(px(18), 'bold', MONO), GREEN, 'center');
    drawBrackets(matCX - px(12), matCY, matW + px(24), matH, GREEN);

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const cx = matCX + c * (cellW + gap);
        const cy = matCY + r * (cellH + gap);
        const cellStepIdx = steps.findIndex(s => s.row === r && s.col === c);
        const isComputed = (t >= 0.6 + (cellStepIdx + 1) * tps);
        const isCurrent = (r === curStep.row && c === curStep.col);
        const resVal = m.result_matrix[r] ? m.result_matrix[r][c] : 0;

        if (isCurrent) {
          const popScale = 1.0 + Math.sin(stepProg * Math.PI) * 0.15;
          withScale(cx + cellW / 2, cy + cellH / 2, popScale, () => {
            panel(cx, cy, cellW, cellH, { r: 12, fill: 'rgba(57, 255, 20, 0.22)', stroke: GREEN, lw: 3, glow: px(28) });
            text(String(resVal), cx + cellW / 2, cy + cellH / 2 + px(10), font(px(30), 'bold', MONO), GREEN, 'center');
          });
        } else if (isComputed) {
          panel(cx, cy, cellW, cellH, { r: 12, fill: 'rgba(20, 60, 35, 0.60)', stroke: 'rgba(57, 255, 20, 0.7)', lw: 2 });
          text(String(resVal), cx + cellW / 2, cy + cellH / 2 + px(9), font(px(28), 'bold', MONO), '#f1f5f9', 'center');
        } else {
          panel(cx, cy, cellW, cellH, { r: 12, fill: 'rgba(10, 15, 30, 0.5)', stroke: 'rgba(51, 65, 85, 0.4)', lw: 1, dash: [4, 4] });
          text('·', cx + cellW / 2, cy + cellH / 2 + px(5), font(px(24), 'bold', MONO), DIM, 'center');
        }
      }
    }
  } else {
    // Vertical layout (Shorts 9:16)
    const gap = px(10);
    const cellW = Math.min(px(90), (box.w - px(80) - (cols - 1) * gap) / cols);
    const cellH = px(64);
    const matW = cols * cellW + (cols - 1) * gap;
    const matH = rows * cellH + (rows - 1) * gap;
    const matX = box.x + (box.w - matW) / 2;

    const topY = box.y + px(10);
    text(`Scalar c = ${m.scalar}  ×  Matrix A:`, box.x + box.w / 2, topY, font(px(22), 'bold', MONO), '#f59e0b', 'center');
    drawBrackets(matX - px(10), topY + px(16), matW + px(20), matH, CYAN);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const cx = matX + c * (cellW + gap), cy = topY + px(16) + r * (cellH + gap);
        const isActive = (r === curStep.row && c === curStep.col);
        const val = m.matrix_a[r] ? m.matrix_a[r][c] : 0;
        panel(cx, cy, cellW, cellH, { r: 10, fill: isActive ? 'rgba(255,230,0,0.2)' : 'rgba(15,23,42,0.9)', stroke: isActive ? YELLOW : 'rgba(100,116,139,0.4)', lw: isActive ? 2.5 : 1 });
        text(String(val), cx + cellW / 2, cy + cellH / 2 + px(8), font(px(24), 'bold', MONO), isActive ? YELLOW : '#cbd5e1', 'center');
      }
    }

    const arrowY = topY + px(24) + matH + px(20);
    text('↓  Multiply each element by scalar  ↓', box.x + box.w / 2, arrowY, font(px(20), 'bold', MONO), MUTED, 'center');

    const botY = arrowY + px(20);
    text('Result Matrix C:', box.x + box.w / 2, botY, font(px(22), 'bold', MONO), GREEN, 'center');
    drawBrackets(matX - px(10), botY + px(16), matW + px(20), matH, GREEN);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const cx = matX + c * (cellW + gap), cy = botY + px(16) + r * (cellH + gap);
        const cellStepIdx = steps.findIndex(s => s.row === r && s.col === c);
        const isComputed = (t >= 0.6 + (cellStepIdx + 1) * tps);
        const isCurrent = (r === curStep.row && c === curStep.col);
        const resVal = m.result_matrix[r] ? m.result_matrix[r][c] : 0;
        panel(cx, cy, cellW, cellH, { r: 10, fill: isCurrent ? 'rgba(57,255,20,0.2)' : isComputed ? 'rgba(20,60,35,0.6)' : 'rgba(10,15,30,0.5)', stroke: isCurrent ? GREEN : isComputed ? GREEN : 'rgba(51,65,85,0.4)', lw: isCurrent ? 2.5 : 1 });
        text(isComputed || isCurrent ? String(resVal) : '·', cx + cellW / 2, cy + cellH / 2 + px(8), font(px(24), 'bold', MONO), isCurrent ? GREEN : isComputed ? '#f1f5f9' : DIM, 'center');
      }
    }
  }

  // Bottom Calculation Panel
  panel(box.x, calcY, box.w, calcH, { r: 16, fill: 'rgba(10, 18, 36, 0.95)', stroke: YELLOW, lw: 2, glow: px(16) });
  const stepBadge = `STEP ${curStepIdx + 1} / ${nSteps}`;
  const sbf = font(px(IS_VERTICAL ? 18 : 20), 'bold', MONO);
  const sbw = measure(stepBadge, sbf) + px(28);
  panel(box.x + px(20), calcY + px(IS_VERTICAL ? 14 : 24), sbw, px(32), { r: 16, fill: 'rgba(255, 230, 0, 0.15)', stroke: YELLOW, lw: 1.5 });
  text(stepBadge, box.x + px(20) + sbw / 2, calcY + px(IS_VERTICAL ? 35 : 45), sbf, YELLOW, 'center');

  const exprStr = curStep.expr || `${m.matrix_a[curStep.row][curStep.col]} × ${m.scalar} = ${curStep.val}`;
  const formulaLabel = `A[${curStep.row}, ${curStep.col}] × ${m.scalar}  =  ${exprStr}`;
  const ef = font(px(IS_VERTICAL ? 22 : 30), 'bold', MONO);
  text(formulaLabel, box.x + px(35) + sbw, calcY + px(IS_VERTICAL ? 36 : 46), ef, '#ffffff');

  const targetText = `→ C[${curStep.row}, ${curStep.col}] = ${curStep.val}`;
  const tf = font(px(IS_VERTICAL ? 18 : 22), 'bold', MONO);
  const tw = measure(targetText, tf) + px(30);
  panel(box.x + box.w - tw - px(20), calcY + px(IS_VERTICAL ? 14 : 24), tw, px(32), { r: 16, fill: 'rgba(57, 255, 20, 0.15)', stroke: GREEN, lw: 1.5 });
  text(targetText, box.x + box.w - tw / 2 - px(20), calcY + px(IS_VERTICAL ? 35 : 45), tf, GREEN, 'center');
};

// --- visual_algo: bảng dữ liệu & tính toán công thức từng bước kiểu Deep-ML ---
TEMPLATES.visual_algo = function (t, box) {
  if (D.matrix_op && !D.visual_algo) {
    return TEMPLATES.matrix_op(t, box);
  }
  const v = D.visual_algo; if (!v || !v.table) return;
  const tbl = v.table;
  const headers = tbl.headers || [];
  const rows = tbl.rows || [];
  const nRows = rows.length;
  if (!headers.length || !nRows) return;

  const availT = Math.max(DURATION - 1.2, 1.0);
  const tps = availT / nRows;
  const activeRowIdx = clamp(Math.floor((t - 0.6) / tps), 0, nRows - 1);

  const fCardH = px(IS_VERTICAL ? 120 : 96);
  const fCardY = box.y + box.h - fCardH;
  const tblH = box.h - fCardH - px(24);

  const nCols = headers.length;
  const colGap = px(10);
  const rowGap = px(10);
  const rowH = Math.min(px(IS_VERTICAL ? 70 : 88), (tblH - px(20) - nRows * rowGap) / (nRows + 1));
  const colW = (box.w - (nCols - 1) * colGap) / nCols;

  const hdrY = box.y + px(10);
  for (let c = 0; c < nCols; c++) {
    const cx = box.x + c * (colW + colGap);
    panel(cx, hdrY, colW, rowH, { r: 10, fill: 'rgba(30, 41, 59, 0.9)', stroke: CYAN, lw: 1.5 });
    text(fitText(headers[c], colW - px(16), font(px(IS_VERTICAL ? 18 : 24), 'bold', FONT)), cx + colW / 2, hdrY + rowH / 2 + px(8), font(px(IS_VERTICAL ? 18 : 24), 'bold', FONT), CYAN, 'center');
  }

  for (let r = 0; r < nRows; r++) {
    const ry = hdrY + (r + 1) * (rowH + rowGap);
    const isActive = (r === activeRowIdx);
    const isPast = (r < activeRowIdx);
    const rowData = rows[r] || [];

    const bgCol = isActive ? 'rgba(255, 230, 0, 0.15)' : isPast ? 'rgba(20, 40, 30, 0.7)' : 'rgba(15, 23, 42, 0.8)';
    const strokeCol = isActive ? YELLOW : isPast ? GREEN : 'rgba(71, 85, 105, 0.4)';
    const textCol = isActive ? YELLOW : isPast ? '#f1f5f9' : MUTED;
    const lw = isActive ? 2.5 : 1;

    for (let c = 0; c < nCols; c++) {
      const cx = box.x + c * (colW + colGap);
      const cellVal = String(rowData[c] != null ? rowData[c] : '');
      panel(cx, ry, colW, rowH, { r: 10, fill: bgCol, stroke: strokeCol, lw: lw, glow: isActive ? px(14) : 0 });
      text(fitText(cellVal, colW - px(16), font(px(IS_VERTICAL ? 18 : 23), 'bold', FONT)), cx + colW / 2, ry + rowH / 2 + px(7), font(px(IS_VERTICAL ? 18 : 23), 'bold', FONT), textCol, 'center');
    }
  }

  panel(box.x, fCardY, box.w, fCardH, { r: 16, fill: 'rgba(10, 18, 36, 0.95)', stroke: GREEN, lw: 2, glow: px(16) });
  const rowTag = `ROW ${activeRowIdx + 1} / ${nRows}`;
  const rtf = font(px(IS_VERTICAL ? 16 : 19), 'bold', FONT);
  const rtw = measure(rowTag, rtf) + px(28);
  panel(box.x + px(20), fCardY + px(IS_VERTICAL ? 14 : 26), rtw, px(38), { r: 16, fill: 'rgba(57, 255, 20, 0.15)', stroke: GREEN, lw: 1.5 });
  text(rowTag, box.x + px(20) + rtw / 2, fCardY + px(IS_VERTICAL ? 37 : 50), rtf, GREEN, 'center');

  const formulaText = v.formula || 'Step-by-step Evaluation';
  const ff = font(px(IS_VERTICAL ? 22 : 30), 'bold', FONT);
  text(formulaText, box.x + px(35) + rtw, fCardY + px(IS_VERTICAL ? 37 : 50), ff, '#ffffff');
};

// --- tree_graph: Cây nhị phân tìm kiếm BST / Cấu trúc dữ liệu cây đồ thị kiểu Deep-ML ---
TEMPLATES.tree_graph = function (t, box) {
  const tg = D.tree_graph; if (!tg) return TEMPLATES.architecture(t, box);
  const nodes = tg.nodes || [
    { id: 0, val: 8 },
    { id: 1, val: 3 }, { id: 2, val: 10 },
    { id: 3, val: 1 }, { id: 4, val: 6 }, { id: 5, val: 9 }, { id: 6, val: 14 }
  ];
  const edges = tg.edges || [
    [0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [2, 6]
  ];
  const steps = tg.steps || [
    { node: 0, status: 'visiting', label: 'Root = 8. So sánh với target: 7 < 8 → Nhánh TRÁI' },
    { node: 1, status: 'visiting', label: 'Node = 3. So sánh với target: 7 > 3 → Nhánh PHẢI' },
    { node: 4, status: 'found', label: 'Node = 6. Tìm thấy vị trí thích hợp!' }
  ];
  const nSteps = steps.length || 1;
  const availT = Math.max(DURATION - 1.2, 1.0);
  const tps = availT / nSteps;
  const curStepIdx = clamp(Math.floor((t - 0.6) / tps), 0, nSteps - 1);
  const curStep = steps[curStepIdx] || { node: 0, status: 'visiting', label: '' };
  const stepProg = clamp((t - 0.6 - curStepIdx * tps) / tps, 0, 1);

  const cardH = px(IS_VERTICAL ? 120 : 96);
  const cardY = box.y + box.h - cardH;
  const treeH = box.h - cardH - px(24);

  const nodePos = {};
  const nNodes = nodes.length;
  for (let i = 0; i < nNodes; i++) {
    const nid = nodes[i].id != null ? nodes[i].id : i;
    let lx, ly;
    if (nodes[i].x != null && nodes[i].y != null) {
      lx = box.x + nodes[i].x * box.w;
      ly = box.y + nodes[i].y * treeH;
    } else {
      const lvl = Math.floor(Math.log2(i + 1));
      const posInLvl = i - (Math.pow(2, lvl) - 1);
      const totalInLvl = Math.pow(2, lvl);
      const stepX = box.w / (totalInLvl + 1);
      lx = box.x + (posInLvl + 1) * stepX;
      ly = box.y + px(IS_VERTICAL ? 44 : 38) + lvl * (treeH - px(IS_VERTICAL ? 90 : 80)) / 2;
    }
    nodePos[nid] = { x: lx, y: ly, val: nodes[i].val };
  }

  const visitedSet = new Set();
  const activeNid = curStep.node != null ? curStep.node : 0;
  for (let s = 0; s <= curStepIdx; s++) {
    if (steps[s] && steps[s].node != null) visitedSet.add(steps[s].node);
  }

  // 1. Edges
  for (const [u, v] of edges) {
    const p1 = nodePos[u], p2 = nodePos[v];
    if (!p1 || !p2) continue;
    const isTraversed = visitedSet.has(u) && visitedSet.has(v);
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    if (isTraversed) {
      ctx.strokeStyle = GREEN;
      ctx.lineWidth = px(3.5);
      ctx.shadowColor = GREEN;
      ctx.shadowBlur = px(12);
    } else {
      ctx.strokeStyle = 'rgba(100, 116, 139, 0.45)';
      ctx.lineWidth = px(2);
    }
    ctx.stroke();
    ctx.restore();
  }

  // 2. Nodes
  const nodeR = px(IS_VERTICAL ? 28 : 34);
  for (let i = 0; i < nNodes; i++) {
    const nid = nodes[i].id != null ? nodes[i].id : i;
    const p = nodePos[nid];
    if (!p) continue;
    const isActive = (nid === activeNid);
    const isVisited = visitedSet.has(nid) && !isActive;

    ctx.save();
    if (isActive) {
      const pulse = 1.0 + Math.sin(stepProg * Math.PI) * 0.15;
      withScale(p.x, p.y, pulse, () => {
        ctx.beginPath(); ctx.arc(p.x, p.y, nodeR + px(8), 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(255, 230, 0, 0.25)'; ctx.fill();
        ctx.beginPath(); ctx.arc(p.x, p.y, nodeR, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(20, 25, 45, 0.95)'; ctx.fill();
        ctx.strokeStyle = YELLOW; ctx.lineWidth = px(3.5);
        ctx.shadowColor = YELLOW; ctx.shadowBlur = px(20); ctx.stroke();
        text(String(p.val), p.x, p.y + px(10), font(px(IS_VERTICAL ? 22 : 28), 'bold', MONO), YELLOW, 'center');
      });
    } else if (isVisited) {
      ctx.beginPath(); ctx.arc(p.x, p.y, nodeR, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(15, 35, 25, 0.95)'; ctx.fill();
      ctx.strokeStyle = GREEN; ctx.lineWidth = px(2.5);
      ctx.shadowColor = GREEN; ctx.shadowBlur = px(12); ctx.stroke();
      text(String(p.val), p.x, p.y + px(9), font(px(IS_VERTICAL ? 20 : 26), 'bold', MONO), '#f1f5f9', 'center');
    } else {
      ctx.beginPath(); ctx.arc(p.x, p.y, nodeR, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(15, 23, 42, 0.85)'; ctx.fill();
      ctx.strokeStyle = 'rgba(100, 116, 139, 0.5)'; ctx.lineWidth = px(1.5); ctx.stroke();
      text(String(p.val), p.x, p.y + px(9), font(px(IS_VERTICAL ? 20 : 26), 'bold', MONO), '#94a3b8', 'center');
    }
    ctx.restore();
  }

  // 3. Bottom Status Panel
  panel(box.x, cardY, box.w, cardH, { r: 16, fill: 'rgba(10, 18, 36, 0.95)', stroke: YELLOW, lw: 2, glow: px(16) });
  const stepBadge = `STEP ${curStepIdx + 1} / ${nSteps}`;
  const sbf = font(px(IS_VERTICAL ? 16 : 19), 'bold', FONT);
  const sbw = measure(stepBadge, sbf) + px(24);
  panel(box.x + px(20), cardY + px(IS_VERTICAL ? 14 : 24), sbw, px(32), { r: 16, fill: 'rgba(255, 230, 0, 0.15)', stroke: YELLOW, lw: 1.5 });
  text(stepBadge, box.x + px(20) + sbw / 2, cardY + px(IS_VERTICAL ? 35 : 45), sbf, YELLOW, 'center');

  const stepLabel = curStep.label || `Duyệt node ${activeNid}`;
  const lf = font(px(IS_VERTICAL ? 20 : 25), 'bold', FONT);
  const maxLabelW = box.w - sbw - px(160);
  text(fitText(stepLabel, maxLabelW, lf), box.x + px(35) + sbw, cardY + px(IS_VERTICAL ? 36 : 46), lf, '#ffffff');

  if (tg.target != null) {
    const tgtText = `Target: ${tg.target}`;
    const tf = font(px(IS_VERTICAL ? 16 : 20), 'bold', MONO);
    const tw = measure(tgtText, tf) + px(26);
    panel(box.x + box.w - tw - px(20), cardY + px(IS_VERTICAL ? 14 : 24), tw, px(32), { r: 16, fill: 'rgba(0, 240, 255, 0.15)', stroke: CYAN, lw: 1.5 });
    text(tgtText, box.x + box.w - tw / 2 - px(20), cardY + px(IS_VERTICAL ? 35 : 45), tf, CYAN, 'center');
  }
};

// --- two_pointers: Hai con trỏ / Cửa sổ trượt Sliding Window kiểu Deep-ML ---
TEMPLATES.two_pointers = function (t, box) {
  const tp = D.two_pointers; if (!tp) return TEMPLATES.architecture(t, box);
  const arr = tp.data || [2, 7, 11, 15];
  const target = tp.target != null ? tp.target : 9;
  const steps = tp.steps || [
    { left: 0, right: arr.length - 1, sum: arr[0] + arr[arr.length - 1], cmp: '>', action: 'Tổng > Target → right--' }
  ];
  const nSteps = steps.length || 1;
  const availT = Math.max(DURATION - 1.2, 1.0);
  const tps = availT / nSteps;
  const curStepIdx = clamp(Math.floor((t - 0.6) / tps), 0, nSteps - 1);
  const curStep = steps[curStepIdx] || { left: 0, right: arr.length - 1, sum: 0, cmp: '', action: '' };

  const cardH = px(IS_VERTICAL ? 120 : 96);
  const cardY = box.y + box.h - cardH;
  const mainH = box.h - cardH - px(24);

  const n = arr.length;
  const gap = px(12);
  const cellW = Math.min(px(IS_VERTICAL ? 80 : 120), (box.w - px(60) - (n - 1) * gap) / n);
  const cellH = px(IS_VERTICAL ? 80 : 100);
  const totalArrW = n * cellW + (n - 1) * gap;
  const arrX = box.x + (box.w - totalArrW) / 2;
  const arrY = box.y + mainH / 2 - cellH / 2 + px(10);

  // 1. Các phần tử mảng và con trỏ
  for (let i = 0; i < n; i++) {
    const cx = arrX + i * (cellW + gap);
    const isL = (i === curStep.left);
    const isR = (i === curStep.right);
    const inWindow = (i >= curStep.left && i <= curStep.right);

    if (isL || isR) {
      const pColor = isL ? CYAN : YELLOW;
      const pLabel = isL && isR ? 'L, R' : (isL ? 'Left' : 'Right');
      const arrowY = arrY - px(24);
      text('▼', cx + cellW / 2, arrowY, font(px(26), 'bold', MONO), pColor, 'center');
      text(pLabel, cx + cellW / 2, arrowY - px(22), font(px(IS_VERTICAL ? 16 : 20), 'bold', MONO), pColor, 'center');
    }

    let fill = 'rgba(15, 23, 42, 0.85)', stroke = 'rgba(100, 116, 139, 0.4)', lw = 1.5, glow = 0;
    if (isL) { fill = 'rgba(0, 240, 255, 0.18)'; stroke = CYAN; lw = 3; glow = px(20); }
    else if (isR) { fill = 'rgba(255, 230, 0, 0.18)'; stroke = YELLOW; lw = 3; glow = px(20); }
    else if (inWindow) { fill = 'rgba(30, 41, 59, 0.7)'; stroke = 'rgba(148, 163, 184, 0.5)'; }

    panel(cx, arrY, cellW, cellH, { r: 14, fill, stroke, lw, glow });
    text(`[${i}]`, cx + cellW / 2, arrY + px(22), font(px(14), 'bold', MONO), MUTED, 'center');
    const valColor = isL ? CYAN : isR ? YELLOW : '#f1f5f9';
    text(String(arr[i]), cx + cellW / 2, arrY + cellH / 2 + px(18), font(px(IS_VERTICAL ? 28 : 38), 'bold', MONO), valColor, 'center');
  }

  // 2. Thẻ tính toán trung tâm
  const valL = arr[curStep.left] != null ? arr[curStep.left] : 0;
  const valR = arr[curStep.right] != null ? arr[curStep.right] : 0;
  const curSum = curStep.sum != null ? curStep.sum : (valL + valR);
  const cmpStr = curStep.cmp || (curSum === target ? '==' : curSum < target ? '<' : '>');
  const mathStr = `arr[${curStep.left}] + arr[${curStep.right}]  =  ${valL} + ${valR}  =  ${curSum}`;

  const calcMidY = arrY + cellH + px(IS_VERTICAL ? 35 : 45);
  text(mathStr, box.x + box.w / 2, calcMidY, font(px(IS_VERTICAL ? 20 : 28), 'bold', MONO), '#ffffff', 'center');

  const matchStr = `Tổng (${curSum})  ${cmpStr}  Target (${target})`;
  const matchCol = cmpStr === '==' ? GREEN : (cmpStr === '<' ? CYAN : YELLOW);
  text(matchStr, box.x + box.w / 2, calcMidY + px(IS_VERTICAL ? 32 : 38), font(px(IS_VERTICAL ? 18 : 24), 'bold', MONO), matchCol, 'center');

  // 3. Bottom Action Panel
  panel(box.x, cardY, box.w, cardH, { r: 16, fill: 'rgba(10, 18, 36, 0.95)', stroke: matchCol, lw: 2, glow: px(16) });
  const stepBadge = `STEP ${curStepIdx + 1} / ${nSteps}`;
  const sbf = font(px(IS_VERTICAL ? 16 : 19), 'bold', FONT);
  const sbw = measure(stepBadge, sbf) + px(24);
  panel(box.x + px(20), cardY + px(IS_VERTICAL ? 14 : 24), sbw, px(32), { r: 16, fill: 'rgba(255, 230, 0, 0.15)', stroke: YELLOW, lw: 1.5 });
  text(stepBadge, box.x + px(20) + sbw / 2, cardY + px(IS_VERTICAL ? 35 : 45), sbf, YELLOW, 'center');

  const actionText = curStep.action || `Xét cặp con trỏ [${curStep.left}, ${curStep.right}]`;
  const af = font(px(IS_VERTICAL ? 20 : 25), 'bold', FONT);
  text(fitText(actionText, box.w - sbw - px(60), af), box.x + px(35) + sbw, cardY + px(IS_VERTICAL ? 36 : 46), af, '#ffffff');
};

// --- stack_queue: Ngăn xếp (Stack LIFO) & Hàng đợi (Queue FIFO) kiểu Deep-ML ---
TEMPLATES.stack_queue = function (t, box) {
  const sq = D.stack_queue; if (!sq) return TEMPLATES.architecture(t, box);
  const kind = (sq.kind || 'stack').toLowerCase();
  const isStack = (kind === 'stack');
  const ops = sq.ops || [
    { op: 'PUSH', val: 42 }, { op: 'PUSH', val: 17 }, { op: 'PUSH', val: 99 },
    { op: 'POP', val: 99 }, { op: 'PUSH', val: 55 }
  ];
  const nOps = ops.length || 1;
  const availT = Math.max(DURATION - 1.2, 1.0);
  const tps = availT / nOps;
  const curStepIdx = clamp(Math.floor((t - 0.6) / tps), 0, nOps - 1);
  const curStep = ops[curStepIdx] || { op: 'PUSH', val: 0 };

  const items = [];
  for (let s = 0; s <= curStepIdx; s++) {
    const o = ops[s];
    if (o.op.toUpperCase() === 'PUSH' || o.op.toUpperCase() === 'ENQUEUE') {
      items.push(o.val);
    } else if (o.op.toUpperCase() === 'POP' || o.op.toUpperCase() === 'DEQUEUE') {
      if (isStack) items.pop();
      else items.shift();
    }
  }

  const cardH = px(IS_VERTICAL ? 120 : 96);
  const cardY = box.y + box.h - cardH;
  const mainH = box.h - cardH - px(24);

  if (isStack) {
    const troughW = Math.min(px(IS_VERTICAL ? 280 : 360), box.w * 0.45);
    const itemH = px(IS_VERTICAL ? 54 : 64);
    const maxItems = 6;
    const troughH = maxItems * itemH + px(40);
    const troughX = box.x + (box.w - troughW) / 2;
    const troughY = box.y + (mainH - troughH) / 2 + px(10);

    ctx.save();
    ctx.strokeStyle = CYAN; ctx.lineWidth = px(3);
    ctx.shadowColor = CYAN; ctx.shadowBlur = px(12);
    ctx.beginPath();
    ctx.moveTo(troughX, troughY);
    ctx.lineTo(troughX, troughY + troughH);
    ctx.lineTo(troughX + troughW, troughY + troughH);
    ctx.lineTo(troughX + troughW, troughY);
    ctx.stroke(); ctx.restore();

    text('STACK (LIFO)', troughX + troughW / 2, troughY - px(16), font(px(IS_VERTICAL ? 20 : 24), 'bold', MONO), CYAN, 'center');
    text('← BASE / ĐÁY', troughX + troughW + px(16), troughY + troughH - px(10), font(px(14), 'bold', MONO), MUTED);

    const padX = px(12);
    const elementW = troughW - padX * 2;
    for (let i = 0; i < items.length && i < maxItems; i++) {
      const ey = troughY + troughH - (i + 1) * itemH - px(6);
      const isTop = (i === items.length - 1);
      const elCol = isTop ? GREEN : '#cbd5e1';
      panel(troughX + padX, ey, elementW, itemH - px(8), {
        r: 10,
        fill: isTop ? 'rgba(57, 255, 20, 0.2)' : 'rgba(30, 41, 59, 0.85)',
        stroke: isTop ? GREEN : 'rgba(100, 116, 139, 0.5)',
        lw: isTop ? 2.5 : 1,
        glow: isTop ? px(16) : 0
      });
      text(String(items[i]), troughX + troughW / 2, ey + itemH / 2 + px(4), font(px(IS_VERTICAL ? 24 : 30), 'bold', MONO), elCol, 'center');

      if (isTop) {
        text('← TOP', troughX + troughW + px(16), ey + itemH / 2 + px(4), font(px(18), 'bold', MONO), GREEN);
      }
    }
  } else {
    const queueW = Math.min(px(IS_VERTICAL ? 540 : 800), box.w - px(80));
    const queueH = px(IS_VERTICAL ? 90 : 110);
    const queueX = box.x + (box.w - queueW) / 2;
    const queueY = box.y + (mainH - queueH) / 2;

    ctx.save();
    ctx.strokeStyle = CYAN; ctx.lineWidth = px(3);
    ctx.shadowColor = CYAN; ctx.shadowBlur = px(12);
    ctx.beginPath();
    ctx.moveTo(queueX, queueY); ctx.lineTo(queueX + queueW, queueY);
    ctx.moveTo(queueX, queueY + queueH); ctx.lineTo(queueX + queueW, queueY + queueH);
    ctx.stroke(); ctx.restore();

    text('← DEQUEUE (Front)', queueX, queueY - px(16), font(px(IS_VERTICAL ? 16 : 20), 'bold', MONO), GREEN);
    text('ENQUEUE (Rear) ←', queueX + queueW, queueY - px(16), font(px(IS_VERTICAL ? 16 : 20), 'bold', MONO), YELLOW, 'right');

    const maxItems = 6;
    const itemW = px(IS_VERTICAL ? 70 : 100);
    const itemGap = px(10);
    for (let i = 0; i < items.length && i < maxItems; i++) {
      const ix = queueX + px(20) + i * (itemW + itemGap);
      const iy = queueY + px(10);
      const isFront = (i === 0);
      panel(ix, iy, itemW, queueH - px(20), {
        r: 10,
        fill: isFront ? 'rgba(57, 255, 20, 0.2)' : 'rgba(30, 41, 59, 0.85)',
        stroke: isFront ? GREEN : 'rgba(100, 116, 139, 0.5)',
        lw: isFront ? 2.5 : 1,
        glow: isFront ? px(14) : 0
      });
      text(String(items[i]), ix + itemW / 2, iy + queueH / 2 - px(2), font(px(IS_VERTICAL ? 22 : 28), 'bold', MONO), isFront ? GREEN : '#cbd5e1', 'center');
    }
  }

  panel(box.x, cardY, box.w, cardH, { r: 16, fill: 'rgba(10, 18, 36, 0.95)', stroke: YELLOW, lw: 2, glow: px(16) });
  const stepBadge = `STEP ${curStepIdx + 1} / ${nOps}`;
  const sbf = font(px(IS_VERTICAL ? 16 : 19), 'bold', FONT);
  const sbw = measure(stepBadge, sbf) + px(24);
  panel(box.x + px(20), cardY + px(IS_VERTICAL ? 14 : 24), sbw, px(32), { r: 16, fill: 'rgba(255, 230, 0, 0.15)', stroke: YELLOW, lw: 1.5 });
  text(stepBadge, box.x + px(20) + sbw / 2, cardY + px(IS_VERTICAL ? 35 : 45), sbf, YELLOW, 'center');

  const opLabel = `Thao tác: ${curStep.op} (${curStep.val})  |  Số phần tử hiện tại: ${items.length}`;
  const of = font(px(IS_VERTICAL ? 20 : 25), 'bold', FONT);
  text(fitText(opLabel, box.w - sbw - px(60), of), box.x + px(35) + sbw, cardY + px(IS_VERTICAL ? 36 : 46), of, '#ffffff');
};

// --- dp_grid: Bảng quy hoạch động (2D DP Table & Transitions) kiểu Deep-ML ---
TEMPLATES.dp_grid = function (t, box) {
  const dp = D.dp_grid; if (!dp) return TEMPLATES.architecture(t, box);
  const rows = dp.rows || 3, cols = dp.cols || 4;
  const totalCells = rows * cols;
  const availT = Math.max(DURATION - 1.2, 1.0);
  const tps = availT / totalCells;
  const curCellIdx = clamp(Math.floor((t - 0.6) / tps), 0, totalCells - 1);
  const activeR = Math.floor(curCellIdx / cols);
  const activeC = curCellIdx % cols;

  const cardH = px(IS_VERTICAL ? 120 : 96);
  const cardY = box.y + box.h - cardH;
  const gridH = box.h - cardH - px(24);

  const grid = dp.grid || [];
  if (!grid.length) {
    for (let r = 0; r < rows; r++) {
      grid[r] = [];
      for (let c = 0; c < cols; c++) {
        grid[r][c] = (r === 0 || c === 0) ? 1 : (grid[r - 1][c] + grid[r][c - 1]);
      }
    }
  }

  const gap = px(10);
  const cellW = Math.min(px(IS_VERTICAL ? 80 : 120), (box.w - px(60) - (cols - 1) * gap) / cols);
  const cellH = Math.min(px(IS_VERTICAL ? 65 : 85), (gridH - px(40) - (rows - 1) * gap) / rows);
  const totalW = cols * cellW + (cols - 1) * gap;
  const totalH = rows * cellH + (rows - 1) * gap;
  const startX = box.x + (box.w - totalW) / 2;
  const startY = box.y + (gridH - totalH) / 2 + px(10);

  // 1. Ô DP Grid
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const cx = startX + c * (cellW + gap);
      const cy = startY + r * (cellH + gap);
      const cellIdx = r * cols + c;
      const isActive = (r === activeR && c === activeC);
      const isCalculated = (cellIdx <= curCellIdx);
      const val = grid[r] ? grid[r][c] : 0;

      let fill = 'rgba(15, 23, 42, 0.75)', stroke = 'rgba(71, 85, 105, 0.4)', lw = 1.5, glow = 0;
      if (isActive) {
        fill = 'rgba(255, 230, 0, 0.22)'; stroke = YELLOW; lw = 3; glow = px(22);
      } else if (isCalculated) {
        fill = 'rgba(20, 50, 35, 0.65)'; stroke = GREEN; lw = 2;
      }

      panel(cx, cy, cellW, cellH, { r: 12, fill, stroke, lw, glow });
      text(`[${r},${c}]`, cx + px(10), cy + px(16), font(px(12), 'bold', MONO), MUTED);

      const valCol = isActive ? YELLOW : (isCalculated ? '#f1f5f9' : DIM);
      const valStr = isCalculated ? String(val) : '·';
      text(valStr, cx + cellW / 2, cy + cellH / 2 + px(8), font(px(IS_VERTICAL ? 24 : 32), 'bold', MONO), valCol, 'center');
    }
  }

  // 2. Mũi tên phụ thuộc
  if (activeR > 0) {
    const ax = startX + activeC * (cellW + gap) + cellW / 2;
    const topY = startY + (activeR - 1) * (cellH + gap) + cellH;
    const botY = startY + activeR * (cellH + gap);
    ctx.save(); ctx.strokeStyle = CYAN; ctx.lineWidth = px(2.5); ctx.beginPath();
    ctx.moveTo(ax, topY + px(2)); ctx.lineTo(ax, botY - px(2)); ctx.stroke();
    text('▼', ax, botY + px(2), font(px(14), 'bold'), CYAN, 'center');
    ctx.restore();
  }
  if (activeC > 0) {
    const ay = startY + activeR * (cellH + gap) + cellH / 2;
    const leftX = startX + (activeC - 1) * (cellW + gap) + cellW;
    const rightX = startX + activeC * (cellW + gap);
    ctx.save(); ctx.strokeStyle = CYAN; ctx.lineWidth = px(2.5); ctx.beginPath();
    ctx.moveTo(leftX + px(2), ay); ctx.lineTo(rightX - px(2), ay); ctx.stroke();
    text('▶', rightX - px(2), ay + px(5), font(px(14), 'bold'), CYAN, 'center');
    ctx.restore();
  }

  // 3. Bottom Calculation Panel
  panel(box.x, cardY, box.w, cardH, { r: 16, fill: 'rgba(10, 18, 36, 0.95)', stroke: YELLOW, lw: 2, glow: px(16) });
  const cellBadge = `CELL [${activeR}, ${activeC}]`;
  const cbf = font(px(IS_VERTICAL ? 16 : 19), 'bold', FONT);
  const cbw = measure(cellBadge, cbf) + px(24);
  panel(box.x + px(20), cardY + px(IS_VERTICAL ? 14 : 24), cbw, px(32), { r: 16, fill: 'rgba(255, 230, 0, 0.15)', stroke: YELLOW, lw: 1.5 });
  text(cellBadge, box.x + px(20) + cbw / 2, cardY + px(IS_VERTICAL ? 35 : 45), cbf, YELLOW, 'center');

  const curVal = grid[activeR] ? grid[activeR][activeC] : 0;
  const topVal = (activeR > 0 && grid[activeR - 1]) ? grid[activeR - 1][activeC] : 0;
  const leftVal = (activeC > 0 && grid[activeR]) ? grid[activeR][activeC - 1] : 0;
  const calcText = (activeR === 0 && activeC === 0) ? `Cơ sở dp[0][0] = ${curVal}` :
                   (activeR === 0) ? `dp[0][${activeC}] = dp[0][${activeC-1}] = ${curVal}` :
                   (activeC === 0) ? `dp[${activeR}][0] = dp[${activeR-1}][0] = ${curVal}` :
                   `dp[${activeR}][${activeC}] = ${topVal} (trên) + ${leftVal} (trái) = ${curVal}`;

  const cf = font(px(IS_VERTICAL ? 20 : 25), 'bold', FONT);
  text(fitText(calcText, box.w - cbw - px(60), cf), box.x + px(35) + cbw, cardY + px(IS_VERTICAL ? 36 : 46), cf, '#ffffff');
};

// --- flowchart: Sơ đồ luồng xử lý / Data Pipeline / Request-Response Flow ---
TEMPLATES.flowchart = function (t, box) {
  const fc = D.flowchart; if (!fc) return TEMPLATES.architecture(t, box);
  const nodes = fc.nodes || [
    { id: 0, label: 'Client App' },
    { id: 1, label: 'API Gateway' },
    { id: 2, label: 'Event Queue' },
    { id: 3, label: 'Worker Service' },
    { id: 4, label: 'Storage & DB' }
  ];
  const conns = fc.connections || [[0, 1], [1, 2], [2, 3], [3, 4]];
  const n = nodes.length;
  if (!n) return TEMPLATES.architecture(t, box);

  const ph = phaseOf(t, n);
  const activeIdx = Math.min(n - 1, Math.floor(ph));
  const cardH = px(IS_VERTICAL ? 120 : 96);
  const cardY = box.y + box.h - cardH;
  const flowH = box.h - cardH - px(24);

  const horiz = !IS_VERTICAL && n <= 5;
  const gap = px(horiz ? 36 : 24);
  const nodeW = horiz ? Math.min(px(240), (box.w - gap * (n - 1)) / n) : Math.min(box.w - px(60), px(550));
  const nodeH = horiz ? Math.min(px(140), flowH * 0.5) : Math.min(px(85), (flowH - gap * (n - 1)) / n);

  const totalW = horiz ? (n * nodeW + (n - 1) * gap) : nodeW;
  const totalH = horiz ? nodeH : (n * nodeH + (n - 1) * gap);
  const startX = box.x + (box.w - totalW) / 2;
  const startY = box.y + (flowH - totalH) / 2;

  const nodePos = i => {
    if (horiz) return { x: startX + i * (nodeW + gap), y: startY };
    return { x: startX, y: startY + i * (nodeH + gap) };
  };

  // 1. Mũi tên & Kết nối
  for (const [u, v] of conns) {
    if (u >= n || v >= n) continue;
    const p1 = nodePos(u), p2 = nodePos(v);
    const x1 = horiz ? p1.x + nodeW : p1.x + nodeW / 2;
    const y1 = horiz ? p1.y + nodeH / 2 : p1.y + nodeH;
    const x2 = horiz ? p2.x : p2.x + nodeW / 2;
    const y2 = horiz ? p2.y + nodeH / 2 : p2.y;

    ctx.save();
    ctx.strokeStyle = 'rgba(0, 240, 255, 0.35)';
    ctx.lineWidth = px(3);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    text(horiz ? '▶' : '▼', horiz ? x2 - px(6) : x2, horiz ? y2 + px(5) : y2 - px(2), font(px(14), 'bold'), CYAN, 'center');
    ctx.restore();

    if (activeIdx === u) {
      const stepProg = clamp(ph - u, 0, 1);
      const packetX = lerp(x1, x2, stepProg);
      const packetY = lerp(y1, y2, stepProg);
      ctx.save();
      ctx.fillStyle = YELLOW;
      ctx.shadowColor = YELLOW;
      ctx.shadowBlur = px(18);
      ctx.beginPath();
      ctx.arc(packetX, packetY, px(7), 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
  }

  // 2. Nodes
  nodes.forEach((nd, i) => {
    const p = nodePos(i);
    const isActive = (i === activeIdx);
    const isPast = (i < activeIdx);
    const e = pop(t, i * 0.15, 0.5);

    withScale(p.x + nodeW / 2, p.y + nodeH / 2, e, () => {
      ctx.save();
      ctx.globalAlpha = e;

      let fill = 'rgba(15, 23, 42, 0.85)', stroke = 'rgba(100, 116, 139, 0.45)', lw = 1.5, glow = 0;
      if (isActive) {
        fill = 'rgba(255, 230, 0, 0.18)'; stroke = YELLOW; lw = 3; glow = px(22);
      } else if (isPast) {
        fill = 'rgba(20, 50, 35, 0.75)'; stroke = GREEN; lw = 2;
      }

      panel(p.x, p.y, nodeW, nodeH, { r: 14, fill, stroke, lw, glow });

      const numStr = String(i + 1).padStart(2, '0');
      const numCol = isActive ? YELLOW : (isPast ? GREEN : CYAN);
      text(numStr, p.x + px(16), p.y + px(22), font(px(14), 'bold', MONO), numCol);

      const maxL = Math.max(2, Math.floor((nodeH - px(34)) / px(horiz ? 24 : 20)));
      const fs = px(horiz ? 22 : 20);
      const th = tbHeight(nd.label, nodeW - px(32), fs, { weight: 'bold', maxLines: maxL });
      tb(nd.label, p.x + px(16), p.y + px(28) + Math.max(0, (nodeH - px(36) - th) / 2), nodeW - px(32), fs, '#f1f5f9', { weight: 'bold', maxLines: maxL });

      ctx.restore();
    });
  });

  // 3. Bottom Status Card
  panel(box.x, cardY, box.w, cardH, { r: 16, fill: 'rgba(10, 18, 36, 0.95)', stroke: YELLOW, lw: 2, glow: px(16) });
  const stepBadge = `STAGE ${activeIdx + 1} / ${n}`;
  const sbf = font(px(IS_VERTICAL ? 16 : 19), 'bold', FONT);
  const sbw = measure(stepBadge, sbf) + px(24);
  panel(box.x + px(20), cardY + px(IS_VERTICAL ? 14 : 24), sbw, px(32), { r: 16, fill: 'rgba(255, 230, 0, 0.15)', stroke: YELLOW, lw: 1.5 });
  text(stepBadge, box.x + px(20) + sbw / 2, cardY + px(IS_VERTICAL ? 35 : 45), sbf, YELLOW, 'center');

  const curLabel = nodes[activeIdx] ? `Đang xử lý tại trạm: ${nodes[activeIdx].label}` : 'Dòng dữ liệu tuần tự';
  const cf = font(px(IS_VERTICAL ? 20 : 25), 'bold', FONT);
  text(fitText(curLabel, box.w - sbw - px(60), cf), box.x + px(35) + sbw, cardY + px(IS_VERTICAL ? 36 : 46), cf, '#ffffff');
};

// --- system_layers: Sơ đồ kiến trúc phân tầng hệ thống (Multi-Tier Architecture) ---
TEMPLATES.system_layers = function (t, box) {
  const sl = D.system_layers; if (!sl) return TEMPLATES.architecture(t, box);
  const layers = sl.layers || [
    { name: 'Presentation Tier', components: ['Web SPA', 'Mobile App', 'Public API'] },
    { name: 'Gateway & Security', components: ['Rate Limiter', 'OAuth2 / JWT', 'Reverse Proxy'] },
    { name: 'Application Logic', components: ['User Service', 'Order Engine', 'Worker Async'] },
    { name: 'Data & Persistence', components: ['Redis Cache', 'PostgreSQL DB', 'Object Storage'] }
  ];
  const n = layers.length;
  if (!n) return TEMPLATES.architecture(t, box);

  const cardH = px(IS_VERTICAL ? 120 : 96);
  const cardY = box.y + box.h - cardH;
  const areaH = box.h - cardH - px(24);

  const gap = px(12);
  const layerH = Math.min(px(IS_VERTICAL ? 90 : 105), (areaH - (n - 1) * gap) / n);
  const totalH = n * layerH + (n - 1) * gap;
  const startY = box.y + (areaH - totalH) / 2;

  const colors = [CYAN, PURPLE, GREEN, YELLOW];

  layers.forEach((lyr, i) => {
    const y = startY + i * (layerH + gap);
    const col = colors[i % colors.length];
    const e = pop(t, i * 0.18, 0.5);

    withScale(box.x + box.w / 2, y + layerH / 2, e, () => {
      ctx.save();
      ctx.globalAlpha = e;

      panel(box.x, y, box.w, layerH, { r: 14, fill: 'rgba(15, 23, 42, 0.88)', stroke: col, lw: 2, glow: px(12) });

      const tagW = Math.min(px(IS_VERTICAL ? 180 : 260), box.w * 0.28);
      panel(box.x + px(10), y + px(8), tagW, layerH - px(16), { r: 10, fill: col + '22', stroke: col, lw: 1.5 });
      text(`0${i+1}`, box.x + px(22), y + px(26), font(px(14), 'bold', MONO), col);
      text(fitText(lyr.name, tagW - px(24), font(px(IS_VERTICAL ? 16 : 21), 'bold', FONT)), box.x + px(22), y + layerH / 2 + px(8), font(px(IS_VERTICAL ? 16 : 21), 'bold', FONT), '#ffffff');

      const comps = lyr.components || [];
      const compAreaX = box.x + tagW + px(24);
      const compAreaW = box.w - tagW - px(34);
      const cGap = px(10);
      const cCount = Math.max(1, comps.length);
      const cW = Math.min(px(IS_VERTICAL ? 150 : 220), (compAreaW - (cCount - 1) * cGap) / cCount);
      const cH = layerH - px(20);

      comps.forEach((cName, ci) => {
        const cx = compAreaX + ci * (cW + cGap);
        const cy = y + px(10);
        panel(cx, cy, cW, cH, { r: 8, fill: 'rgba(30, 41, 59, 0.9)', stroke: 'rgba(148, 163, 184, 0.4)', lw: 1 });
        text(fitText(cName, cW - px(16), font(px(IS_VERTICAL ? 15 : 20), 'bold', FONT)), cx + cW / 2, cy + cH / 2 + px(6), font(px(IS_VERTICAL ? 15 : 20), 'bold', FONT), '#cbd5e1', 'center');
      });

      ctx.restore();
    });
  });

  // Bottom Status Card
  panel(box.x, cardY, box.w, cardH, { r: 16, fill: 'rgba(10, 18, 36, 0.95)', stroke: CYAN, lw: 2, glow: px(16) });
  const badgeText = `TIERS: ${n} LAYERS`;
  const sbf = font(px(IS_VERTICAL ? 16 : 19), 'bold', FONT);
  const sbw = measure(badgeText, sbf) + px(24);
  panel(box.x + px(20), cardY + px(IS_VERTICAL ? 14 : 24), sbw, px(32), { r: 16, fill: 'rgba(0, 240, 255, 0.15)', stroke: CYAN, lw: 1.5 });
  text(badgeText, box.x + px(20) + sbw / 2, cardY + px(IS_VERTICAL ? 35 : 45), sbf, CYAN, 'center');

  const desc = 'Kiến trúc phân tầng độc lập: Tách rời trách nhiệm (Decoupled) & Dễ mở rộng';
  const df = font(px(IS_VERTICAL ? 20 : 25), 'bold', FONT);
  text(fitText(desc, box.w - sbw - px(60), df), box.x + px(35) + sbw, cardY + px(IS_VERTICAL ? 36 : 46), df, '#ffffff');
};

// --- comparison_matrix: Bảng biểu ma trận so sánh đặc tính kỹ thuật ---
TEMPLATES.comparison_matrix = function (t, box) {
  const cm = D.comparison_matrix; if (!cm) return TEMPLATES.architecture(t, box);
  const headers = cm.headers || ['Tiêu chí', 'Giải pháp A (Naive)', 'Giải pháp B (Tối ưu)'];
  const rows = cm.rows || [
    ['Độ phức tạp thời gian', 'O(N²)', 'O(N log N) ✔'],
    ['Bộ nhớ phụ trợ', 'O(N)', 'O(1) ✔'],
    ['Khả năng mở rộng (Scale)', 'Kém ✖', 'Cực tốt ✔'],
    ['Độ trễ P99', '450 ms', '12 ms ✔']
  ];
  const nCols = headers.length;
  const nRows = rows.length;
  if (!nCols || !nRows) return TEMPLATES.architecture(t, box);

  const cardH = px(IS_VERTICAL ? 120 : 108);
  const cardY = box.y + box.h - cardH;
  const tblAreaH = box.h - cardH - px(24);

  const colGap = px(10);
  const rowGap = px(10);
  const colW = (box.w - (nCols - 1) * colGap) / nCols;
  const maxRowH = px(IS_VERTICAL ? 76 : 95);
  const rowH = Math.min(maxRowH, Math.max(px(IS_VERTICAL ? 55 : 70), (tblAreaH - px(20) - nRows * rowGap) / (nRows + 1)));

  const startY = box.y + px(10);

  // 1. Header Row
  for (let c = 0; c < nCols; c++) {
    const cx = box.x + c * (colW + colGap);
    const isWinner = (c === nCols - 1 && nCols > 1);
    const hCol = isWinner ? GREEN : CYAN;
    panel(cx, startY, colW, rowH, { r: 12, fill: isWinner ? 'rgba(57,255,20,0.18)' : 'rgba(30,41,59,0.92)', stroke: hCol, lw: 2, glow: isWinner ? px(14) : 0 });
    text(fitText(headers[c], colW - px(20), font(px(IS_VERTICAL ? 18 : 25), 'bold', FONT)), cx + colW / 2, startY + rowH / 2 + px(8), font(px(IS_VERTICAL ? 18 : 25), 'bold', FONT), hCol, 'center');
  }

  // 2. Data Rows
  for (let r = 0; r < nRows; r++) {
    const ry = startY + (r + 1) * (rowH + rowGap);
    const rowData = rows[r] || [];
    const e = pop(t, 0.2 + r * 0.15, 0.4);

    for (let c = 0; c < nCols; c++) {
      const cx = box.x + c * (colW + colGap);
      const rawVal = String(rowData[c] != null ? rowData[c] : '');
      const isWinnerCol = (c === nCols - 1 && nCols > 1);
      const isHeaderCol = (c === 0);

      const hasCheck = rawVal.includes('✔');
      const hasCross = rawVal.includes('✖');

      let textCol = '#f1f5f9';
      let stroke = 'rgba(71, 85, 105, 0.45)';
      let fill = isHeaderCol ? 'rgba(15, 23, 42, 0.95)' : isWinnerCol ? 'rgba(20, 50, 35, 0.65)' : 'rgba(15, 23, 42, 0.80)';

      if (hasCheck) textCol = GREEN;
      else if (hasCross) textCol = PINK;
      else if (isWinnerCol) textCol = '#ffffff';
      else if (isHeaderCol) textCol = '#e2e8f0';

      if (isWinnerCol) stroke = 'rgba(57, 255, 20, 0.55)';

      ctx.save();
      ctx.globalAlpha = e;
      panel(cx, ry, colW, rowH, { r: 10, fill, stroke, lw: isWinnerCol ? 2 : 1.2 });
      text(fitText(rawVal, colW - px(20), font(px(IS_VERTICAL ? 17 : 23), 'bold', FONT)), cx + colW / 2, ry + rowH / 2 + px(7), font(px(IS_VERTICAL ? 17 : 23), 'bold', FONT), textCol, 'center');
      ctx.restore();
    }
  }

  // 3. Bottom Card
  panel(box.x, cardY, box.w, cardH, { r: 16, fill: 'rgba(10, 18, 36, 0.95)', stroke: GREEN, lw: 2, glow: px(16) });
  const bText = 'BENCHMARK SUMMARY';
  const sbf = font(px(IS_VERTICAL ? 17 : 20), 'bold', FONT);
  const sbw = measure(bText, sbf) + px(28);
  panel(box.x + px(20), cardY + px(IS_VERTICAL ? 14 : 24), sbw, px(IS_VERTICAL ? 36 : 40), { r: 16, fill: 'rgba(57, 255, 20, 0.15)', stroke: GREEN, lw: 1.5 });
  text(bText, box.x + px(20) + sbw / 2, cardY + px(IS_VERTICAL ? 37 : 49), sbf, GREEN, 'center');

  const winnerName = headers[headers.length - 1] || 'Giải pháp đề xuất';
  const note = `Đánh giá tổng quan: ${winnerName} vượt trội về hiệu năng và khả năng mở rộng.`;
  const nf = font(px(IS_VERTICAL ? 20 : 25), 'bold', FONT);
  text(fitText(note, box.w - sbw - px(60), nf), box.x + px(35) + sbw, cardY + px(IS_VERTICAL ? 37 : 49), nf, '#ffffff');
};

// --- radar_chart: Biểu đồ Radar mạng nhện so sánh đa tiêu chí (Trade-off Analysis) ---
TEMPLATES.radar_chart = function (t, box) {
  const rc = D.radar_chart; if (!rc) return TEMPLATES.architecture(t, box);
  const axes = rc.axes || ['Tốc độ', 'Bộ nhớ', 'Mở rộng', 'Tin cậy', 'Chi phí'];
  const series = rc.series || [
    { name: 'Truyền thống', values: [40, 80, 45, 85, 90], color: PINK },
    { name: 'Hiện đại', values: [95, 85, 90, 70, 60], color: CYAN }
  ];
  const numAxes = axes.length;
  if (numAxes < 3) return TEMPLATES.architecture(t, box);

  const cardH = px(IS_VERTICAL ? 120 : 96);
  const cardY = box.y + box.h - cardH;
  const radarH = box.h - cardH - px(24);

  const cx = box.x + box.w / 2;
  const cy = box.y + radarH / 2;
  const maxR = Math.min(box.w * 0.38, radarH * 0.40);

  const angleStep = (Math.PI * 2) / numAxes;

  // 1. Vòng lưới đa giác đồng tâm (Concentric Grid)
  const gridLevels = [0.25, 0.50, 0.75, 1.0];
  ctx.save();
  gridLevels.forEach(lvl => {
    ctx.strokeStyle = 'rgba(100, 116, 139, 0.3)';
    ctx.lineWidth = px(1);
    ctx.beginPath();
    for (let i = 0; i < numAxes; i++) {
      const a = i * angleStep - Math.PI / 2;
      const x = cx + Math.cos(a) * maxR * lvl;
      const y = cy + Math.sin(a) * maxR * lvl;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.stroke();
  });

  // 2. Các tia trục nối từ tâm & Nhãn tiêu chí
  for (let i = 0; i < numAxes; i++) {
    const a = i * angleStep - Math.PI / 2;
    const ax = cx + Math.cos(a) * maxR;
    const ay = cy + Math.sin(a) * maxR;

    ctx.strokeStyle = 'rgba(100, 116, 139, 0.4)';
    ctx.lineWidth = px(1.5);
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(ax, ay);
    ctx.stroke();

    const labelDist = maxR + px(IS_VERTICAL ? 28 : 36);
    const lx = cx + Math.cos(a) * labelDist;
    const ly = cy + Math.sin(a) * labelDist + px(6);
    const af = font(px(IS_VERTICAL ? 17 : 22), 'bold', FONT);
    text(axes[i], lx, ly, af, '#cbd5e1', 'center');
  }
  ctx.restore();

  // 3. Đa giác dữ liệu Series
  series.forEach((s, sIdx) => {
    const sColor = s.color || (sIdx === 0 ? PINK : CYAN);
    const vals = s.values || [];
    const animProg = easeOutExpo(clamp((t - 0.3 - sIdx * 0.3) / 1.0, 0, 1));

    ctx.save();
    ctx.beginPath();
    for (let i = 0; i < numAxes; i++) {
      const a = i * angleStep - Math.PI / 2;
      const val = (vals[i] != null ? vals[i] : 50) / 100;
      const r = maxR * val * animProg;
      const x = cx + Math.cos(a) * r;
      const y = cy + Math.sin(a) * r;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fillStyle = sColor + '28';
    ctx.fill();
    ctx.strokeStyle = sColor;
    ctx.lineWidth = px(2.5);
    ctx.shadowColor = sColor;
    ctx.shadowBlur = px(12);
    ctx.stroke();

    for (let i = 0; i < numAxes; i++) {
      const a = i * angleStep - Math.PI / 2;
      const val = (vals[i] != null ? vals[i] : 50) / 100;
      const r = maxR * val * animProg;
      const x = cx + Math.cos(a) * r;
      const y = cy + Math.sin(a) * r;
      ctx.beginPath();
      ctx.arc(x, y, px(5), 0, Math.PI * 2);
      ctx.fillStyle = sColor;
      ctx.fill();
    }
    ctx.restore();
  });

  // 4. Legend
  const legY = box.y + px(10);
  let legX = box.x + px(20);
  series.forEach((s, sIdx) => {
    const sColor = s.color || (sIdx === 0 ? PINK : CYAN);
    const lf = font(px(18), 'bold', FONT);
    const lw = measure(s.name, lf) + px(40);
    panel(legX, legY, lw, px(30), { r: 14, fill: sColor + '18', stroke: sColor, lw: 1.5 });
    ctx.fillStyle = sColor; ctx.beginPath(); ctx.arc(legX + px(15), legY + px(15), px(5), 0, Math.PI * 2); ctx.fill();
    text(s.name, legX + px(28), legY + px(21), lf, '#ffffff');
    legX += lw + px(12);
  });

  // 5. Bottom Card
  panel(box.x, cardY, box.w, cardH, { r: 16, fill: 'rgba(10, 18, 36, 0.95)', stroke: YELLOW, lw: 2, glow: px(16) });
  const bText = 'TRADE-OFF RADAR';
  const sbf = font(px(IS_VERTICAL ? 16 : 19), 'bold', FONT);
  const sbw = measure(bText, sbf) + px(24);
  panel(box.x + px(20), cardY + px(IS_VERTICAL ? 14 : 24), sbw, px(32), { r: 16, fill: 'rgba(255, 230, 0, 0.15)', stroke: YELLOW, lw: 1.5 });
  text(bText, box.x + px(20) + sbw / 2, cardY + px(IS_VERTICAL ? 35 : 45), sbf, YELLOW, 'center');

  const tradeOffMsg = 'Không có giải pháp hoàn hảo: Luôn là sự đánh đổi giữa các tiêu chí.';
  const tf = font(px(IS_VERTICAL ? 20 : 25), 'bold', FONT);
  text(fitText(tradeOffMsg, box.w - sbw - px(60), tf), box.x + px(35) + sbw, cardY + px(IS_VERTICAL ? 36 : 46), tf, '#ffffff');
};

// --- trie_tree: Cây tiền tố Prefix Tree / Trie (String Matching & Autocomplete) ---
TEMPLATES.trie_tree = function (t, box) {
  const tt = D.trie_tree; if (!tt) return TEMPLATES.architecture(t, box);
  const words = tt.words || ['cat', 'car', 'cart', 'dog'];
  const query = (tt.query || 'car').toLowerCase();
  const qLen = query.length;

  const cardH = px(IS_VERTICAL ? 120 : 96);
  const cardY = box.y + box.h - cardH;
  const treeH = box.h - cardH - px(24);

  const availT = Math.max(DURATION - 1.2, 1.0);
  const tps = availT / (qLen + 1);
  const activeCharIdx = clamp(Math.floor((t - 0.6) / tps), 0, qLen);

  // Cấu trúc node cây Trie trực quan
  const nodes = [
    { id: 'root', label: 'ROOT', level: 0, xFrac: 0.5 },
    { id: 'c', label: 'c', level: 1, xFrac: 0.35, parent: 'root' },
    { id: 'd', label: 'd', level: 1, xFrac: 0.68, parent: 'root' },
    { id: 'ca', label: 'a', level: 2, xFrac: 0.35, parent: 'c' },
    { id: 'do', label: 'o', level: 2, xFrac: 0.68, parent: 'd' },
    { id: 'cat', label: 't', level: 3, xFrac: 0.24, parent: 'ca', isWord: true },
    { id: 'car', label: 'r', level: 3, xFrac: 0.44, parent: 'ca', isWord: true },
    { id: 'dog', label: 'g', level: 3, xFrac: 0.68, parent: 'do', isWord: true },
    { id: 'cart', label: 't', level: 4, xFrac: 0.44, parent: 'car', isWord: true },
  ];

  const levelY = lvl => box.y + px(IS_VERTICAL ? 36 : 30) + lvl * (treeH - px(IS_VERTICAL ? 80 : 70)) / 4;
  const nodeMap = {};
  nodes.forEach(nd => {
    nodeMap[nd.id] = {
      x: box.x + nd.xFrac * box.w,
      y: levelY(nd.level),
      nd
    };
  });

  // Đường đi tìm kiếm query (ví dụ: 'root' -> 'c' -> 'ca' -> 'car')
  const searchPath = ['root'];
  let curPrefix = '';
  for (let i = 0; i < qLen; i++) {
    curPrefix += query[i];
    searchPath.push(curPrefix);
  }

  // 1. Vẽ các cạnh (Edges)
  nodes.forEach(nd => {
    if (!nd.parent) return;
    const p1 = nodeMap[nd.parent], p2 = nodeMap[nd.id];
    if (!p1 || !p2) return;

    const isPathEdge = searchPath.includes(nd.parent) && searchPath.includes(nd.id) &&
                       searchPath.indexOf(nd.id) <= activeCharIdx;

    ctx.save();
    ctx.strokeStyle = isPathEdge ? GREEN : 'rgba(100, 116, 139, 0.4)';
    ctx.lineWidth = isPathEdge ? px(3.5) : px(2);
    if (isPathEdge) { ctx.shadowColor = GREEN; ctx.shadowBlur = px(12); }
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.stroke();
    ctx.restore();
  });

  // 2. Vẽ các Node
  const nodeR = px(IS_VERTICAL ? 22 : 28);
  nodes.forEach(nd => {
    const p = nodeMap[nd.id];
    if (!p) return;
    const pathIdx = searchPath.indexOf(nd.id);
    const isTraversed = (pathIdx >= 0 && pathIdx <= activeCharIdx);
    const isCurrent = (pathIdx === activeCharIdx);

    ctx.save();
    let fill = 'rgba(15, 23, 42, 0.85)', stroke = 'rgba(100, 116, 139, 0.45)', textCol = '#94a3b8';
    if (isCurrent) {
      fill = 'rgba(255, 230, 0, 0.22)'; stroke = YELLOW; textCol = YELLOW;
      ctx.shadowColor = YELLOW; ctx.shadowBlur = px(18);
    } else if (isTraversed) {
      fill = 'rgba(20, 50, 35, 0.85)'; stroke = GREEN; textCol = '#f1f5f9';
      ctx.shadowColor = GREEN; ctx.shadowBlur = px(10);
    }

    ctx.beginPath();
    ctx.arc(p.x, p.y, nodeR, 0, Math.PI * 2);
    ctx.fillStyle = fill; ctx.fill();
    ctx.strokeStyle = stroke; ctx.lineWidth = isCurrent ? px(3) : isTraversed ? px(2.5) : px(1.5); ctx.stroke();

    const fs = px(nd.id === 'root' ? 14 : (IS_VERTICAL ? 18 : 22));
    text(nd.label, p.x, p.y + px(fs * 0.35), font(fs, 'bold', MONO), textCol, 'center');

    // Huy hiệu kết thúc từ (End of Word)
    if (nd.isWord) {
      ctx.beginPath(); ctx.arc(p.x + nodeR * 0.7, p.y - nodeR * 0.7, px(5), 0, Math.PI * 2);
      ctx.fillStyle = isTraversed ? GREEN : CYAN; ctx.fill();
    }
    ctx.restore();
  });

  // 3. Bottom Card
  panel(box.x, cardY, box.w, cardH, { r: 16, fill: 'rgba(10, 18, 36, 0.95)', stroke: YELLOW, lw: 2, glow: px(16) });
  const stepBadge = `SEARCH: "${query.slice(0, activeCharIdx)}"`;
  const sbf = font(px(IS_VERTICAL ? 16 : 18), 'bold', MONO);
  const sbw = measure(stepBadge, sbf) + px(24);
  panel(box.x + px(20), cardY + px(IS_VERTICAL ? 14 : 24), sbw, px(32), { r: 16, fill: 'rgba(255, 230, 0, 0.15)', stroke: YELLOW, lw: 1.5 });
  text(stepBadge, box.x + px(20) + sbw / 2, cardY + px(IS_VERTICAL ? 35 : 45), sbf, YELLOW, 'center');

  const isComplete = (activeCharIdx >= qLen);
  const statusMsg = isComplete ? `Tìm kiếm từ "${query}": Thành công! Đã khớp tới node kết thúc.` : `Đang duyệt theo tiền tố ký tự '${query[activeCharIdx - 1] || 'root'}'`;
  const sf = font(px(IS_VERTICAL ? 18 : 22), 'bold', MONO);
  text(fitText(statusMsg, box.w - sbw - px(60), sf), box.x + px(35) + sbw, cardY + px(IS_VERTICAL ? 36 : 46), sf, isComplete ? GREEN : '#ffffff');
};

// --- graph_network: Đồ thị có trọng số & Thuật toán Dijkstra (Weighted Graph & Shortest Path) ---
TEMPLATES.graph_network = function (t, box) {
  const gn = D.graph_network; if (!gn) return TEMPLATES.architecture(t, box);
  const rawNodes = (gn.nodes || ['A', 'B', 'C', 'D', 'E']).map(n => {
    if (typeof n === 'object' && n !== null) return { id: String(n.id || n.name || n.label), label: String(n.label || n.id || n.name) };
    return { id: String(n), label: String(n) };
  });
  const edges = (gn.edges || [['A', 'B', 4], ['A', 'C', 2], ['C', 'B', 1], ['B', 'D', 5], ['C', 'D', 8], ['D', 'E', 3]]).map(e => {
    if (Array.isArray(e)) return [String(e[0]), String(e[1]), e[2] != null ? e[2] : 1];
    if (typeof e === 'object' && e !== null) return [String(e.from || e.u || e.source), String(e.to || e.v || e.target), e.weight != null ? e.weight : (e.w || 1)];
    return ['', '', 1];
  }).filter(e => e[0] && e[1]);
  const path = (gn.path || ['A', 'C', 'B', 'D', 'E']).map(p => (typeof p === 'object' && p !== null) ? String(p.id || p.name || p.label) : String(p));

  const cardH = px(IS_VERTICAL ? 120 : 96);
  const cardY = box.y + box.h - cardH;
  const graphH = box.h - cardH - px(24);

  const numNodes = rawNodes.length;
  const cx = box.x + box.w / 2, cy = box.y + graphH / 2;
  const radius = Math.min(box.w * 0.38, graphH * 0.40);

  const nodePos = {};
  rawNodes.forEach((nd, i) => {
    const a = (i * Math.PI * 2) / numNodes - Math.PI / 2;
    nodePos[nd.id] = { x: cx + Math.cos(a) * radius, y: cy + Math.sin(a) * radius, label: nd.label };
  });

  const pathEdges = new Set();
  for (let i = 0; i < path.length - 1; i++) {
    pathEdges.add(`${path[i]}-${path[i+1]}`);
    pathEdges.add(`${path[i+1]}-${path[i]}`);
  }

  const pAnim = clamp((t - 0.4) / (DURATION - 1.2), 0, 1);
  const activePathStep = Math.min(path.length - 1, Math.floor(pAnim * path.length));

  // 1. Vẽ các cạnh có trọng số
  edges.forEach(([u, v, w]) => {
    const p1 = nodePos[u], p2 = nodePos[v];
    if (!p1 || !p2) return;
    const isShortest = pathEdges.has(`${u}-${v}`);

    ctx.save();
    ctx.strokeStyle = isShortest ? GREEN : 'rgba(100, 116, 139, 0.4)';
    ctx.lineWidth = isShortest ? px(3.5) : px(1.8);
    if (isShortest) { ctx.shadowColor = GREEN; ctx.shadowBlur = px(12); }
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.stroke();

    // Weight Badge ở giữa cạnh
    const mx = (p1.x + p2.x) / 2, my = (p1.y + p2.y) / 2;
    panel(mx - px(14), my - px(12), px(28), px(24), { r: 6, fill: 'rgba(15, 23, 42, 0.95)', stroke: isShortest ? GREEN : 'rgba(71, 85, 105, 0.5)', lw: 1 });
    text(String(w), mx, my + px(5), font(px(14), 'bold', MONO), isShortest ? GREEN : '#cbd5e1', 'center');
    ctx.restore();
  });

  // 2. Vẽ các Node
  const nodeR = px(IS_VERTICAL ? 24 : 30);
  rawNodes.forEach(nd => {
    const p = nodePos[nd.id];
    if (!p) return;
    const pathIdx = path.indexOf(nd.id);
    const inShortest = (pathIdx >= 0 && pathIdx <= activePathStep);
    const isCurrent = (pathIdx === activePathStep);

    ctx.save();
    let fill = 'rgba(15, 23, 42, 0.85)', stroke = 'rgba(100, 116, 139, 0.5)', textCol = '#94a3b8';
    if (isCurrent) {
      fill = 'rgba(255, 230, 0, 0.22)'; stroke = YELLOW; textCol = YELLOW;
      ctx.shadowColor = YELLOW; ctx.shadowBlur = px(20);
    } else if (inShortest) {
      fill = 'rgba(20, 50, 35, 0.85)'; stroke = GREEN; textCol = '#f1f5f9';
      ctx.shadowColor = GREEN; ctx.shadowBlur = px(12);
    }

    ctx.beginPath();
    ctx.arc(p.x, p.y, nodeR, 0, Math.PI * 2);
    ctx.fillStyle = fill; ctx.fill();
    ctx.strokeStyle = stroke; ctx.lineWidth = isCurrent ? px(3.5) : inShortest ? px(2.5) : px(1.5); ctx.stroke();

    text(nd.id, p.x, p.y + px(9), font(px(IS_VERTICAL ? 20 : 24), 'bold', MONO), textCol, 'center');

    if (nd.label && nd.label !== nd.id) {
      const lblY = p.y >= cy ? p.y + nodeR + px(20) : p.y - nodeR - px(10);
      text(nd.label, p.x, lblY, font(px(IS_VERTICAL ? 13 : 15), 'bold'), isCurrent ? YELLOW : (inShortest ? '#f1f5f9' : MUTED), 'center');
    }
    ctx.restore();
  });

  // 3. Bottom Card
  panel(box.x, cardY, box.w, cardH, { r: 16, fill: 'rgba(10, 18, 36, 0.95)', stroke: GREEN, lw: 2, glow: px(16) });
  const gnData = D.graph_network || {};
  const bText = gnData.badge || (gnData.is_dijkstra ? 'DIJKSTRA ALGO' : 'GRAPH FLOW');
  const sbf = font(px(IS_VERTICAL ? 16 : 18), 'bold', MONO);
  const sbw = measure(bText, sbf) + px(24);
  panel(box.x + px(20), cardY + px(IS_VERTICAL ? 14 : 24), sbw, px(32), { r: 16, fill: 'rgba(57, 255, 20, 0.15)', stroke: GREEN, lw: 1.5 });
  text(bText, box.x + px(20) + sbw / 2, cardY + px(IS_VERTICAL ? 35 : 45), sbf, GREEN, 'center');

  const pStr = path.slice(0, activePathStep + 1).join(' → ');
  const pathPrefix = gnData.is_dijkstra ? 'Đường đi ngắn nhất: ' : 'Luồng phân tán: ';
  const pathMsg = `${pathPrefix}${pStr}`;
  const pf = font(px(IS_VERTICAL ? 18 : 22), 'bold', MONO);
  text(fitText(pathMsg, box.w - sbw - px(60), pf), box.x + px(35) + sbw, cardY + px(IS_VERTICAL ? 36 : 46), pf, '#ffffff');
};

// --- donut_chart: Biểu đồ tròn Donut Cyberpunk với phân khúc phát sáng & Explode ---
TEMPLATES.donut_chart = function (t, box) {
  const dc = D.donut_chart; if (!dc) return TEMPLATES.architecture(t, box);
  const labels = dc.labels || ['Transformer', 'CNN', 'RNN', 'MLP'];
  const values = dc.values || [55, 25, 12, 8];
  const unit = dc.unit || '%';
  const total = values.reduce((a, b) => a + b, 0) || 100;
  const n = values.length;

  const cardH = px(IS_VERTICAL ? 120 : 96);
  const cardY = box.y + box.h - cardH;
  const chartH = box.h - cardH - px(24);

  const cx = box.x + (IS_VERTICAL ? box.w / 2 : box.w * 0.38);
  const cy = box.y + (IS_VERTICAL ? chartH * 0.42 : chartH / 2);
  const maxR = Math.min(box.w * (IS_VERTICAL ? 0.35 : 0.28), chartH * 0.40);
  const innerR = maxR * 0.58;

  const colors = [CYAN, PURPLE, GREEN, YELLOW];
  const animProg = easeOutExpo(clamp(t / 1.2, 0, 1));

  // 1. Vẽ các lát cắt Donut
  let startAngle = -Math.PI / 2;
  values.forEach((v, i) => {
    const sliceAngle = (v / total) * Math.PI * 2 * animProg;
    const endAngle = startAngle + sliceAngle;
    const col = colors[i % colors.length];

    // Lát cắt lớn nhất bung nhẹ ra ngoài
    const isLargest = (i === 0);
    const midA = startAngle + sliceAngle / 2;
    const explodeDist = isLargest ? px(10) : 0;
    const ox = Math.cos(midA) * explodeDist;
    const oy = Math.sin(midA) * explodeDist;

    ctx.save();
    ctx.translate(ox, oy);
    ctx.beginPath();
    ctx.arc(cx, cy, maxR, startAngle, endAngle);
    ctx.arc(cx, cy, innerR, endAngle, startAngle, true);
    ctx.closePath();
    ctx.fillStyle = col + (isLargest ? 'dd' : '99');
    ctx.shadowColor = col;
    ctx.shadowBlur = isLargest ? px(16) : px(8);
    ctx.fill();
    ctx.strokeStyle = col;
    ctx.lineWidth = px(2);
    ctx.stroke();
    ctx.restore();

    startAngle = endAngle;
  });

  // 2. Chữ số ở tâm Donut
  const mainVal = Math.round(values[0] * animProg);
  text(`${mainVal}${unit}`, cx, cy + px(10), font(px(IS_VERTICAL ? 32 : 44), 'bold', MONO), '#ffffff', 'center');
  text(labels[0], cx, cy - px(IS_VERTICAL ? 26 : 34), font(px(IS_VERTICAL ? 14 : 18), 'bold', MONO), CYAN, 'center');

  // 3. Legend List bên cạnh
  const legX = IS_VERTICAL ? box.x + px(20) : box.x + box.w * 0.68;
  const legY = IS_VERTICAL ? box.y + chartH * 0.72 : box.y + px(30);
  const legW = IS_VERTICAL ? box.w - px(40) : box.w * 0.28;
  const itemH = px(IS_VERTICAL ? 36 : 46);

  labels.forEach((l, i) => {
    const col = colors[i % colors.length];
    const iy = legY + i * itemH;
    ctx.save();
    ctx.fillStyle = col; ctx.beginPath(); ctx.arc(legX + px(10), iy + px(12), px(6), 0, Math.PI * 2); ctx.fill();
    text(l, legX + px(26), iy + px(16), font(px(IS_VERTICAL ? 15 : 18), 'bold', MONO), '#cbd5e1');
    const vStr = `${values[i]}${unit}`;
    text(vStr, legX + legW, iy + px(16), font(px(IS_VERTICAL ? 15 : 18), 'bold', MONO), col, 'right');
    ctx.restore();
  });

  // 4. Bottom Card
  panel(box.x, cardY, box.w, cardH, { r: 16, fill: 'rgba(10, 18, 36, 0.95)', stroke: CYAN, lw: 2, glow: px(16) });
  const bText = 'SHARE ANALYSIS';
  const sbf = font(px(IS_VERTICAL ? 16 : 18), 'bold', MONO);
  const sbw = measure(bText, sbf) + px(24);
  panel(box.x + px(20), cardY + px(IS_VERTICAL ? 14 : 24), sbw, px(32), { r: 16, fill: 'rgba(0, 240, 255, 0.15)', stroke: CYAN, lw: 1.5 });
  text(bText, box.x + px(20) + sbw / 2, cardY + px(IS_VERTICAL ? 35 : 45), sbf, CYAN, 'center');

  const desc = `Thị phần áp đảo: ${labels[0]} chiếm ${values[0]}% tổng tỷ trọng.`;
  const df = font(px(IS_VERTICAL ? 18 : 22), 'bold', MONO);
  text(fitText(desc, box.w - sbw - px(60), df), box.x + px(35) + sbw, cardY + px(IS_VERTICAL ? 36 : 46), df, '#ffffff');
};

// --- math_equation: Phương trình toán học chi tiết với biến số Color-Coded (3Blue1Brown Style) ---
TEMPLATES.math_equation = function (t, box) {
  const me = D.math_equation; if (!me) return TEMPLATES.architecture(t, box);
  const formula = me.formula || 'Attention(Q, K, V) = Softmax( (Q · K^T) / √d_k ) · V';
  const parts = me.parts || [
    { symbol: 'Q · K^T', name: 'Độ tương đồng (Score)', desc: 'Tích vô hướng đo tương quan', color: CYAN },
    { symbol: '√d_k', name: 'Scaling Factor', desc: 'Ổn định gradient khi chiều lớn', color: YELLOW },
    { symbol: 'Softmax', name: 'Chuẩn hoá xác suất', desc: 'Chuyển điểm số về tổng 1.0', color: PINK },
    { symbol: 'V', name: 'Value Matrix', desc: 'Tổng hợp ngữ nghĩa có trọng số', color: GREEN }
  ];

  const cardH = px(IS_VERTICAL ? 120 : 96);
  const cardY = box.y + box.h - cardH;
  const eqAreaH = box.h - cardH - px(24);

  // 1. Khung hiển thị công thức lớn ở trên
  const fBoxH = px(IS_VERTICAL ? 110 : 130);
  const fBoxY = box.y + px(16);
  panel(box.x, fBoxY, box.w, fBoxH, { r: 16, fill: 'rgba(10, 18, 36, 0.95)', stroke: YELLOW, lw: 2.5, glow: px(20) });

  const drawn = drawMath(formula, box.x + box.w / 2, fBoxY + fBoxH / 2, box.w - px(60), fBoxH - px(24), '#ffffff', {
    targetH: px(IS_VERTICAL ? 55 : 75),
    glow: px(12),
    glowColor: 'rgba(255, 230, 0, 0.45)'
  });
  if (!drawn) {
    const ff = font(px(IS_VERTICAL ? 22 : 36), 'bold', MONO);
    text(fitText(formula, box.w - px(60), ff), box.x + box.w / 2, fBoxY + fBoxH / 2 + px(10), ff, '#ffffff', 'center');
  }

  // 2. 4 thẻ giải thích từng thành phần (Callout Cards)
  const nParts = parts.length;
  const cAreaY = fBoxY + fBoxH + px(20);
  const cAreaH = eqAreaH - fBoxH - px(30);

  const horiz = !IS_VERTICAL;
  const gap = px(12);
  const cW = horiz ? (box.w - (nParts - 1) * gap) / nParts : box.w;
  const cH = horiz ? cAreaH : (cAreaH - (nParts - 1) * gap) / nParts;

  parts.forEach((p, i) => {
    const cx = horiz ? box.x + i * (cW + gap) : box.x;
    const cy = horiz ? cAreaY : cAreaY + i * (cH + gap);
    const pCol = p.color || PALETTE[i % PALETTE.length];
    const e = pop(t, 0.4 + i * 0.18, 0.5);

    withScale(cx + cW / 2, cy + cH / 2, e, () => {
      ctx.save();
      ctx.globalAlpha = e;
      panel(cx, cy, cW, cH, { r: 12, fill: 'rgba(15, 23, 42, 0.90)', stroke: pCol, lw: 2, glow: px(12) });

      // Ký hiệu toán học lớn
      const mathDrawn = drawMath(p.symbol, cx + px(16), cy + px(IS_VERTICAL ? 24 : 32), cW - px(32), px(IS_VERTICAL ? 24 : 32), pCol, {
        align: 'left',
        targetH: px(IS_VERTICAL ? 20 : 26)
      });
      if (!mathDrawn) {
        text(p.symbol, cx + px(16), cy + px(IS_VERTICAL ? 24 : 32), font(px(IS_VERTICAL ? 18 : 24), 'bold', MONO), pCol);
      }
      // Tên đại lượng
      text(fitText(p.name, cW - px(32), font(px(IS_VERTICAL ? 15 : 18), 'bold', MONO)), cx + px(16), cy + px(IS_VERTICAL ? 46 : 64), font(px(IS_VERTICAL ? 15 : 18), 'bold', MONO), '#ffffff');
      // Mô tả bản chất
      if (!IS_VERTICAL || cH > px(70)) {
        tb(p.desc, cx + px(16), cy + px(IS_VERTICAL ? 66 : 94), cW - px(32), px(14), '#94a3b8', { maxLines: 2 });
      }
      ctx.restore();
    });
  });

  // 3. Bottom Card
  panel(box.x, cardY, box.w, cardH, { r: 16, fill: 'rgba(10, 18, 36, 0.95)', stroke: YELLOW, lw: 2, glow: px(16) });
  const bText = 'ESSENCE BREAKDOWN';
  const sbf = font(px(IS_VERTICAL ? 16 : 18), 'bold', MONO);
  const sbw = measure(bText, sbf) + px(24);
  panel(box.x + px(20), cardY + px(IS_VERTICAL ? 14 : 24), sbw, px(32), { r: 16, fill: 'rgba(255, 230, 0, 0.15)', stroke: YELLOW, lw: 1.5 });
  text(bText, box.x + px(20) + sbw / 2, cardY + px(IS_VERTICAL ? 35 : 45), sbf, YELLOW, 'center');

  const sumMsg = 'Bản chất toán học: Mỗi ký hiệu đều đảm nhiệm một vai trò then chốt trong mô hình.';
  const sf = font(px(IS_VERTICAL ? 18 : 22), 'bold', MONO);
  text(fitText(sumMsg, box.w - sbw - px(60), sf), box.x + px(35) + sbw, cardY + px(IS_VERTICAL ? 36 : 46), sf, '#ffffff');
};

// --- linear_transform: Biến đổi tuyến tính lưới 2D (Essence of Linear Algebra - 3Blue1Brown) ---
TEMPLATES.linear_transform = function (t, box) {
  const lt = D.linear_transform; if (!lt) return TEMPLATES.architecture(t, box);
  const matrix = lt.matrix || [[2, 1], [0.5, 1.5]];
  const det = lt.det != null ? lt.det : 2.5;
  const ihat = lt.ihat || [matrix[0][0], matrix[1][0]];
  const jhat = lt.jhat || [matrix[0][1], matrix[1][1]];

  const cardH = px(IS_VERTICAL ? 120 : 96);
  const cardY = box.y + box.h - cardH;
  const gridH = box.h - cardH - px(24);

  const cx = box.x + box.w / 2, cy = box.y + gridH / 2;
  const unitSize = px(IS_VERTICAL ? 65 : 85);

  const lambda = easeOutExpo(clamp((t - 0.5) / (DURATION - 1.2), 0, 1));
  // Ma trận nội suy M(t) = (1 - lambda) * I + lambda * M
  const m00 = 1 * (1 - lambda) + matrix[0][0] * lambda;
  const m01 = 0 * (1 - lambda) + matrix[0][1] * lambda;
  const m10 = 0 * (1 - lambda) + matrix[1][0] * lambda;
  const m11 = 1 * (1 - lambda) + matrix[1][1] * lambda;

  const transform = (gx, gy) => {
    const tx = (m00 * gx + m01 * gy) * unitSize;
    const ty = -(m10 * gx + m11 * gy) * unitSize;
    return { x: cx + tx, y: cy + ty };
  };

  // 1. Vẽ các đường lưới toạ độ sau khi biến đổi
  ctx.save();
  ctx.strokeStyle = 'rgba(56, 189, 248, 0.20)';
  ctx.lineWidth = px(1.2);
  const gridRange = 4;

  for (let gx = -gridRange; gx <= gridRange; gx++) {
    const pTop = transform(gx, gridRange), pBot = transform(gx, -gridRange);
    ctx.beginPath(); ctx.moveTo(pTop.x, pTop.y); ctx.lineTo(pBot.x, pBot.y); ctx.stroke();
  }
  for (let gy = -gridRange; gy <= gridRange; gy++) {
    const pLeft = transform(-gridRange, gy), pRight = transform(gridRange, gy);
    ctx.beginPath(); ctx.moveTo(pLeft.x, pLeft.y); ctx.lineTo(pRight.x, pRight.y); ctx.stroke();
  }

  // 2. Hình bình hành đơn vị (Unit Area = det)
  const p00 = transform(0, 0), p10 = transform(1, 0), p11 = transform(1, 1), p01 = transform(0, 1);
  ctx.beginPath();
  ctx.moveTo(p00.x, p00.y); ctx.lineTo(p10.x, p10.y); ctx.lineTo(p11.x, p11.y); ctx.lineTo(p01.x, p01.y);
  ctx.closePath();
  ctx.fillStyle = 'rgba(255, 230, 0, 0.18)'; ctx.fill();
  ctx.strokeStyle = YELLOW; ctx.lineWidth = px(2); ctx.stroke();

  // 3. Vector i-hat (GREEN) và j-hat (RED/PINK)
  const drawVector = (target, col, label) => {
    ctx.strokeStyle = col; ctx.lineWidth = px(3.5); ctx.shadowColor = col; ctx.shadowBlur = px(12);
    ctx.beginPath(); ctx.moveTo(p00.x, p00.y); ctx.lineTo(target.x, target.y); ctx.stroke();
    text('▶', target.x, target.y + px(5), font(px(16), 'bold'), col, 'center');
    text(label, target.x + px(12), target.y - px(8), font(px(16), 'bold', MONO), col);
  };
  drawVector(p10, GREEN, 'i-hat');
  drawVector(p01, PINK, 'j-hat');
  ctx.restore();

  // 4. Matrix Panel ở góc trên
  const matBoxW = px(IS_VERTICAL ? 220 : 280), matBoxH = px(70);
  panel(box.x + px(20), box.y + px(16), matBoxW, matBoxH, { r: 12, fill: 'rgba(10, 18, 36, 0.95)', stroke: CYAN, lw: 1.5 });
  text(`M = [ [${matrix[0][0]}, ${matrix[0][1]}], [${matrix[1][0]}, ${matrix[1][1]} ] ]`, box.x + px(35), box.y + px(42), font(px(15), 'bold', MONO), CYAN);
  text(`det(M) = ${det}`, box.x + px(35), box.y + px(68), font(px(15), 'bold', MONO), YELLOW);

  // 5. Bottom Card
  panel(box.x, cardY, box.w, cardH, { r: 16, fill: 'rgba(10, 18, 36, 0.95)', stroke: YELLOW, lw: 2, glow: px(16) });
  const bText = 'LINEAR TRANSFORMATION';
  const sbf = font(px(IS_VERTICAL ? 16 : 18), 'bold', MONO);
  const sbw = measure(bText, sbf) + px(24);
  panel(box.x + px(20), cardY + px(IS_VERTICAL ? 14 : 24), sbw, px(32), { r: 16, fill: 'rgba(255, 230, 0, 0.15)', stroke: YELLOW, lw: 1.5 });
  text(bText, box.x + px(20) + sbw / 2, cardY + px(IS_VERTICAL ? 35 : 45), sbf, YELLOW, 'center');

  const lMsg = `Bản chất ma trận: Uốn cong không gian toạ độ (i-hat → [${ihat}], j-hat → [${jhat}]).`;
  const lf = font(px(IS_VERTICAL ? 18 : 22), 'bold', MONO);
  text(fitText(lMsg, box.w - sbw - px(60), lf), box.x + px(35) + sbw, cardY + px(IS_VERTICAL ? 36 : 46), lf, '#ffffff');
};

// --- calculus_curve: Giải tích đạo hàm & tiếp tuyến trượt (Essence of Calculus - 3Blue1Brown) ---
TEMPLATES.calculus_curve = function (t, box) {
  const cc = D.calculus_curve; if (!cc) return TEMPLATES.architecture(t, box);
  const funcStr = cc.func_str || 'f(x) = x³ - 2x + 1';

  const cardH = px(IS_VERTICAL ? 120 : 96);
  const cardY = box.y + box.h - cardH;
  const plotH = box.h - cardH - px(24);

  const cx = box.x + box.w / 2, cy = box.y + plotH / 2;
  const scaleX = px(IS_VERTICAL ? 80 : 120), scaleY = px(IS_VERTICAL ? 60 : 80);

  // Hàm số f(x) = 0.5 * x^3 - 1.5 * x
  const f = x => 0.4 * Math.pow(x, 3) - 1.4 * x;
  // Đạo hàm f'(x) = 1.2 * x^2 - 1.4
  const df = x => 1.2 * Math.pow(x, 2) - 1.4;

  const toScreen = (x, y) => ({ x: cx + x * scaleX, y: cy - y * scaleY });

  // 1. Vẽ trục Oxy
  ctx.save();
  ctx.strokeStyle = 'rgba(100, 116, 139, 0.4)';
  ctx.lineWidth = px(1.5);
  ctx.beginPath();
  ctx.moveTo(box.x + px(20), cy); ctx.lineTo(box.x + box.w - px(20), cy);
  ctx.moveTo(cx, box.y + px(10)); ctx.lineTo(cx, box.y + plotH - px(10));
  ctx.stroke();

  // 2. Vẽ đường cong hàm số f(x)
  ctx.strokeStyle = CYAN;
  ctx.lineWidth = px(3.5);
  ctx.shadowColor = CYAN;
  ctx.shadowBlur = px(14);
  ctx.beginPath();
  const xMin = -2.2, xMax = 2.2, steps = 100;
  for (let i = 0; i <= steps; i++) {
    const x = xMin + (i / steps) * (xMax - xMin);
    const p = toScreen(x, f(x));
    if (i === 0) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  }
  ctx.stroke();

  // 3. Điểm tiếp xúc P(x0, f(x0)) trượt theo thời gian t
  const animT = (t / DURATION) * Math.PI * 2;
  const x0 = Math.sin(animT) * 1.5;
  const y0 = f(x0);
  const slope = df(x0);
  const p0 = toScreen(x0, y0);

  // Đường tiếp tuyến (Tangent Line)
  const tanLen = 1.2;
  const t1 = toScreen(x0 - tanLen, y0 - slope * tanLen);
  const t2 = toScreen(x0 + tanLen, y0 + slope * tanLen);
  ctx.strokeStyle = YELLOW;
  ctx.lineWidth = px(3);
  ctx.shadowColor = YELLOW;
  ctx.shadowBlur = px(16);
  ctx.beginPath();
  ctx.moveTo(t1.x, t1.y); ctx.lineTo(t2.x, t2.y);
  ctx.stroke();

  // Điểm P
  ctx.beginPath(); ctx.arc(p0.x, p0.y, px(7), 0, Math.PI * 2);
  ctx.fillStyle = YELLOW; ctx.fill();
  ctx.restore();

  // 4. Slope Indicator Panel ở góc trên
  const statBoxW = px(IS_VERTICAL ? 240 : 300), statBoxH = px(70);
  panel(box.x + px(20), box.y + px(16), statBoxW, statBoxH, { r: 12, fill: 'rgba(10, 18, 36, 0.95)', stroke: YELLOW, lw: 1.5 });
  text(funcStr, box.x + px(35), box.y + px(42), font(px(15), 'bold', MONO), CYAN);
  text(`x = ${x0.toFixed(2)}  |  Hệ số góc f'(x) = ${slope.toFixed(2)}`, box.x + px(35), box.y + px(68), font(px(14), 'bold', MONO), YELLOW);

  // 5. Bottom Card
  panel(box.x, cardY, box.w, cardH, { r: 16, fill: 'rgba(10, 18, 36, 0.95)', stroke: YELLOW, lw: 2, glow: px(16) });
  const bText = 'CALCULUS ESSENCE';
  const sbf = font(px(IS_VERTICAL ? 16 : 19), 'bold', FONT);
  const sbw = measure(bText, sbf) + px(24);
  panel(box.x + px(20), cardY + px(IS_VERTICAL ? 14 : 24), sbw, px(32), { r: 16, fill: 'rgba(255, 230, 0, 0.15)', stroke: YELLOW, lw: 1.5 });
  text(bText, box.x + px(20) + sbw / 2, cardY + px(IS_VERTICAL ? 35 : 45), sbf, YELLOW, 'center');

  const cMsg = 'Đạo hàm: Tốc độ biến thiên tức thời là hệ số góc tiếp tuyến khi khoảng cách tiến về 0.';
  const cf = font(px(IS_VERTICAL ? 20 : 25), 'bold', FONT);
  text(fitText(cMsg, box.w - sbw - px(60), cf), box.x + px(35) + sbw, cardY + px(IS_VERTICAL ? 36 : 46), cf, '#ffffff');
};

// ---------- điều phối ----------
function renderFrameAt(f) {
  const t = Math.min(f / FPS, DURATION);
  drawBackground(t);
  const box = contentBox(drawHeader(t));
  (TEMPLATES[TEMPLATE] || TEMPLATES.architecture)(t, box);
  if (CFG.illustrative) drawBadge(box);
}
function report(kind, msg) { try { return fetch('/' + kind + '?msg=' + encodeURIComponent(String(msg).slice(0, 300)), { method: 'POST' }); } catch (e) { return null; } }
async function prepare() {
  try { if (document.fonts) { if (CFG.custom_font) { await document.fonts.load('20px MGFont'); await document.fonts.load('bold 20px MGFont'); await document.fonts.load('600 20px MGFont'); await document.fonts.load('20px "Be Vietnam Pro"'); await document.fonts.load('bold 20px "Be Vietnam Pro"'); } await document.fonts.ready; } } catch (e) {}
  if (stockImg && stockImg.decode) { try { await stockImg.decode(); } catch (e) {} }
  for (const im of Object.values(MASCOT_IMGS)) {
    if (im && im.decode) { try { await im.decode(); } catch (e) {} }
  }
  try {
    if (window.MathJax && MathJax.tex2svg) {
      if (D.math_equation) {
        const me = D.math_equation;
        if (me.formula) await getMathImage(me.formula, '#ffffff');
        if (Array.isArray(me.parts)) {
          for (const p of me.parts) {
            if (p.symbol) await getMathImage(p.symbol, p.color || CYAN);
          }
        }
      }
      if (D.calculus_curve && D.calculus_curve.func_str) {
        await getMathImage(D.calculus_curve.func_str, CYAN);
      }
    }
  } catch (e) {}
  HL = null;
  const probes = [0, Math.floor(TOTAL_FRAMES / 2), TOTAL_FRAMES - 1];
  for (const f of probes) {
    try { renderFrameAt(f); } catch (e) {
      if (TEMPLATE === 'architecture') throw e;
      await report('warn', 'template ' + TEMPLATE + ' lỗi: ' + e); TEMPLATE = 'architecture'; SORT_STEPS = null; HL = null; renderFrameAt(f); break;
    }
  }
}
async function postFrame(f, blob) {
  for (let a = 0; a < 3; a++) {
    try { const r = await fetch('/frame?idx=' + f, { method: 'POST', body: blob }); if (r.ok) return; } catch (e) {}
    await new Promise(r => setTimeout(r, 100 * (a + 1)));
  }
  throw new Error('không gửi được frame ' + f);
}
async function runCapture() {
  try {
    await prepare();
    const inflight = new Set();
    for (let f = 0; f < TOTAL_FRAMES; f++) {
      renderFrameAt(f);
      const blob = await new Promise(res => canvas.toBlob(res, 'image/jpeg', 0.92));
      const p = postFrame(f, blob).finally(() => inflight.delete(p)); inflight.add(p);
      if (inflight.size >= 6) await Promise.race(inflight);
    }
    await Promise.all(inflight);
    await fetch('/done', { method: 'POST' });
  } catch (e) { await report('error', e && e.message ? e.message : e); }
}
window.renderFrameAt = renderFrameAt; window.prepare = prepare;
const PREVIEW = (String(location.hash).match(/preview=([\d.]+)/) || [])[1];
if (!window.__NO_AUTOSTART__) {
  if (PREVIEW !== undefined) {
    prepare().then(() => {
      renderFrameAt(Math.round(parseFloat(PREVIEW) * FPS));
      document.title = 'ready';
    });
  } else {
    runCapture();
  }
}
"""


def _font_face_css() -> tuple[str, bool]:
    root = Path(__file__).resolve().parent.parent
    fonts_dir = root / "assets" / "fonts"

    def resolve(val) -> Path | None:
        if not val:
            return None
        p = Path(str(val))
        if not p.is_absolute():
            p = root / p
        return p if p.is_file() else None

    bold_p = resolve(_mg_cfg().get("font_file")) or resolve(CONFIG.get("visual", {}).get("font")) or resolve(fonts_dir / "BeVietnamPro-Bold.ttf")
    reg_p = resolve(CONFIG.get("visual", {}).get("font_regular")) or resolve(fonts_dir / "BeVietnamPro-Regular.ttf")
    semi_p = resolve(fonts_dir / "BeVietnamPro-SemiBold.ttf")

    if not bold_p and not reg_p and not semi_p:
        return "", False

    rules = []

    def add_font_rule(p: Path, weight: str):
        try:
            data = base64.b64encode(p.read_bytes()).decode("ascii")
            ext = p.suffix.lower().lstrip(".")
            fmt = {"ttf": "truetype", "otf": "opentype", "woff": "woff", "woff2": "woff2"}.get(ext, "truetype")
            for fam in ("MGFont", "Be Vietnam Pro"):
                rules.append(
                    f"@font-face {{ font-family: '{fam}'; font-weight: {weight}; "
                    f"src: url(data:font/{ext};base64,{data}) format('{fmt}'); font-display: block; }}"
                )
        except OSError as e:
            log.debug("Lỗi đọc font %s: %s", p, e)

    if reg_p:
        add_font_rule(reg_p, "400")
    if semi_p:
        add_font_rule(semi_p, "600")
    if bold_p:
        add_font_rule(bold_p, "700 900")
    elif reg_p:
        add_font_rule(reg_p, "700 900")

    if not rules:
        return "", False
    return "\n".join(rules), True


def _read_image_data_url(image_path: Path | None) -> str:
    if not image_path:
        return ""
    try:
        if image_path.exists():
            raw = image_path.read_bytes()
            if len(raw) > 1000:
                ext = "png" if image_path.suffix.lower() == ".png" else "jpeg"
                return f"data:image/{ext};base64," + base64.b64encode(raw).decode("ascii")
    except OSError as e:
        log.debug("Không thể đọc ảnh minh hoạ: %s", e)
    return ""


def _json_for_script(obj: Any) -> str:
    """JSON an toàn để nhúng trong <script>: không thể thoát khỏi thẻ script (kể cả </script>, <!--)."""
    s = json.dumps(obj, ensure_ascii=False)
    return s.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


_BUNNY_GIRL_CACHE: dict[str, str] | None = None


def _get_bunny_girl_assets() -> dict[str, str]:
    global _BUNNY_GIRL_CACHE
    if _BUNNY_GIRL_CACHE is not None:
        return _BUNNY_GIRL_CACHE
    assets = {}
    m_dir = _ROOT / "assets" / "mascots" / "bunny_girl"
    if m_dir.is_dir():
        for p in m_dir.glob("*.png"):
            try:
                raw = p.read_bytes()
                assets[p.stem] = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
            except Exception as e:
                log.debug("Load mascot asset %s lỗi: %s", p.name, e)
    _BUNNY_GIRL_CACHE = assets
    return assets


def _build_motion_html(
    scene: Scene,
    duration: float,
    width: int,
    height: int,
    fps: int,
    image_path: Path | None = None,
    vendor_url: str | None = None,
) -> str:
    """Tạo trang HTML Canvas cho 1 scene (template + dữ liệu lấy từ chính scene)."""
    lang = "vi" if str(CONFIG.get("language", "vi")).lower().startswith("vi") else "en"
    img = _read_image_data_url(image_path)
    plan = _plan_scene(scene, bool(img), lang)
    subtitle_src = _sentences(scene.narration, 1, 140)
    css, custom_font = _font_face_css()

    vendor_js = (_ROOT / "assets" / "vendor" / "tex-svg.js").resolve()
    mathjax_tag = ""
    if vendor_js.is_file():
        v_url = vendor_url or vendor_js.as_uri()
        mathjax_tag = f'<script src="{v_url}"></script>'

    cfg = {
        "width": int(width), "height": int(height), "fps": int(fps), "duration": float(duration),
        "template": plan["template"], "data": plan["data"], "illustrative": plan["illustrative"],
        "title": _clean_title(scene.heading, lang),
        "subtitle": subtitle_src[0] if subtitle_src else "",
        "lang": lang, "labels": _LABELS[lang], "image": img, "custom_font": custom_font,
        "mascot_images": _get_bunny_girl_assets(),
    }
    html = (_HTML.replace("__LANG__", lang)
            .replace("__FONTFACE__", css)
            .replace("__WIDTH__", str(int(width)))
            .replace("__HEIGHT__", str(int(height)))
            .replace("__MATHJAX_SCRIPT__", mathjax_tag)
            .replace("__JS__", _JS)
            .replace("__CFG__", _json_for_script(cfg)))  # CFG thay SAU CÙNG: dữ liệu người dùng không bị quét lại
    return html


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _kill_tree(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    try:
        if os.name == "nt":
            kw = {"capture_output": True, "timeout": 10}
            if hasattr(subprocess, "CREATE_NO_WINDOW"):
                kw["creationflags"] = subprocess.CREATE_NO_WINDOW
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], **kw)
        else:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
        proc.wait(timeout=5)
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


def _browser_cmd(browser_bin: str, profile: str, width: int, height: int, url: str, extra: list[str] | None = None) -> list[str]:
    cmd = [
        browser_bin, "--headless=new", "--disable-gpu", "--no-sandbox", "--disable-dev-shm-usage",
        f"--user-data-dir={profile}", "--no-first-run", "--no-default-browser-check", "--disable-extensions",
        "--disable-background-timer-throttling", "--disable-renderer-backgrounding",
        "--disable-backgrounding-occluded-windows", "--force-device-scale-factor=1",
        "--allow-file-access-from-files",
        f"--window-size={width},{height}", "--hide-scrollbars",
    ]
    cmd += [str(a) for a in (_mg_cfg().get("extra_args") or [])]
    cmd += extra or []
    cmd.append(url)
    return cmd


def _popen_kwargs() -> dict:
    kw: dict = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if os.name == "nt":
        if hasattr(subprocess, "CREATE_NO_WINDOW"):
            kw["creationflags"] = subprocess.CREATE_NO_WINDOW
    else:
        kw["start_new_session"] = True
    return kw


def render_motion_graphic_scene(
    scene: Scene,
    out_dir: Path,
    duration: float,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    timeout: int = 120,
    image_path: Path | None = None,
) -> Path | None:
    """Render 1 scene thành MP4 bằng headless Edge/Chrome + Canvas + FFmpeg.

    Trả về Path tới out.mp4 nếu ĐỦ 100% frame và FFmpeg thành công, ngược lại None.
    """
    browser_bin = get_browser_path()
    if not browser_bin:
        log.warning("Không tìm thấy Microsoft Edge/Google Chrome để render Motion Graphics.")
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    shutil.rmtree(frames_dir, ignore_errors=True)  # tránh ghép lẫn frame cũ của lần chạy trước
    frames_dir.mkdir(parents=True, exist_ok=True)

    total_expected = max(1, round(duration * fps))
    vendor_js = (_ROOT / "assets" / "vendor" / "tex-svg.js").resolve()
    vendor_url = "/vendor/tex-svg.js" if vendor_js.is_file() else None
    html_bytes = _build_motion_html(scene, duration, width, height, fps, image_path=image_path, vendor_url=vendor_url).encode("utf-8")
    template = re.search(r'"template": "([a-z_]+)"', html_bytes.decode("utf-8", "ignore"))
    log.info("Motion Graphics template '%s' cho scene: %s", template.group(1) if template else "?", (scene.heading or "")[:40])

    captured: set[int] = set()
    lock = threading.Lock()
    done, failed = threading.Event(), threading.Event()
    state = {"last": time.time(), "error": ""}

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def handle(self):
            try:
                super().handle()
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                pass

        def _reply(self, code: int = 200, body: bytes = b"ok", ctype: str = "text/plain") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            p = urlparse(self.path).path
            if p in ("/", "/index.html"):
                self._reply(200, html_bytes, "text/html; charset=utf-8")
            elif p.startswith("/vendor/"):
                vf = (_ROOT / "assets" / p.lstrip("/")).resolve()
                if vf.is_file() and str(vf).startswith(str((_ROOT / "assets" / "vendor").resolve())):
                    self._reply(200, vf.read_bytes(), "application/javascript")
                else:
                    self._reply(404, b"Not found")
            else:
                self._reply(204, b"")

        def do_POST(self):  # noqa: N802
            u = urlparse(self.path)
            length = int(self.headers.get("Content-Length", 0) or 0)
            data = self.rfile.read(length) if length else b""
            q = parse_qs(u.query)
            if u.path == "/frame":
                try:
                    idx = int(q.get("idx", ["-1"])[0])
                except ValueError:
                    idx = -1
                if 0 <= idx < total_expected and data:
                    (frames_dir / f"frame_{idx:05d}.jpg").write_bytes(data)
                    with lock:
                        captured.add(idx)
                        state["last"] = time.time()
                        if len(captured) >= total_expected:
                            done.set()
            elif u.path == "/done":
                done.set()
            elif u.path == "/error":
                state["error"] = q.get("msg", [""])[0]
                failed.set()
            elif u.path == "/warn":
                log.warning("Motion Graphics (trình duyệt): %s", q.get("msg", [""])[0])
            self._reply()

        def log_message(self, *a, **k):  # noqa: D401
            pass

    class Server(http.server.ThreadingHTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    server = Server(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    profile = tempfile.mkdtemp(prefix="mg_profile_")
    cmd = _browser_cmd(browser_bin, profile, width, height, f"http://127.0.0.1:{port}/")
    # Thời gian cho phép tỉ lệ với số frame (render phần mềm ~0.1s/frame ở 1080p), tối thiểu theo tham số
    budget = max(int(timeout), int(_mg_cfg().get("min_timeout", 0) or 0), 30 + int(total_expected * 0.12))
    t0 = time.time()
    proc: subprocess.Popen | None = None
    reason = ""
    try:
        proc = subprocess.Popen(cmd, **_popen_kwargs())
        while True:
            if done.wait(0.5):
                break
            if failed.is_set():
                reason = f"lỗi từ trang: {state['error']}"
                break
            if proc.poll() is not None and not done.is_set():
                time.sleep(0.5)  # có thể /done vừa tới
                if not done.is_set():
                    reason = f"trình duyệt thoát sớm (code {proc.returncode})"
                break
            now = time.time()
            if now - t0 > budget:
                reason = f"vượt timeout {budget}s"
                break
            if now - state["last"] > max(30, budget * 0.5) and now - t0 > 30:
                reason = "không còn frame mới (treo)"
                break
    except Exception as e:  # noqa: BLE001
        reason = f"khởi chạy trình duyệt lỗi: {e}"
    finally:
        _kill_tree(proc)
        try:
            server.shutdown()
            server.server_close()
        except Exception:  # noqa: BLE001
            pass
        shutil.rmtree(profile, ignore_errors=True)

    with lock:
        n_cap = len(captured)
        complete = all(i in captured for i in range(total_expected))
    log.info("Motion Graphics: thu %d/%d frame trong %.1fs", n_cap, total_expected, time.time() - t0)
    keep = bool(_mg_cfg().get("keep_frames"))
    if not complete:
        log.warning("Motion Graphics thất bại: thiếu frame (%d/%d)%s", n_cap, total_expected, f" - {reason}" if reason else "")
        if not keep:
            shutil.rmtree(frames_dir, ignore_errors=True)
        return None

    # --- FFmpeg ---
    out_mp4 = out_dir / "out.mp4"
    ffmpeg_bin = get_ffmpeg_path()
    if not ffmpeg_bin:
        log.warning("Không tìm thấy FFmpeg thực thi trên hệ thống để ghép video.")
        if not keep:
            shutil.rmtree(frames_dir, ignore_errors=True)
        return None

    ff_cmd = [
        ffmpeg_bin, "-y", "-framerate", str(fps), "-i", str(frames_dir / "frame_%05d.jpg"),
        "-frames:v", str(total_expected), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-an",
        "-movflags", "+faststart", str(out_mp4),
    ]
    try:
        kw = {"capture_output": True, "text": True, "timeout": max(60, int(duration * 4))}
        if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
            kw["creationflags"] = subprocess.CREATE_NO_WINDOW
        res = subprocess.run(ff_cmd, **kw)
        ok = res.returncode == 0 and out_mp4.exists() and out_mp4.stat().st_size > 1024
        if ok:
            log.info("Motion Graphics render OK: %s (%.1fMB)", out_mp4.name, out_mp4.stat().st_size / 1e6)
            if not keep:
                shutil.rmtree(frames_dir, ignore_errors=True)
            return out_mp4
        log.warning("FFmpeg ghép MP4 lỗi: %s", (res.stderr or "")[-400:])
    except Exception as e:  # noqa: BLE001
        log.warning("Lỗi khi chạy FFmpeg: %s", e)
    if not keep:
        shutil.rmtree(frames_dir, ignore_errors=True)
    return None


def render_preview_png(
    scene: Scene,
    out_png: Path,
    t: float = 3.0,
    duration: float = 8.0,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    image_path: Path | None = None,
    timeout: int = 60,
) -> Path | None:
    """Chụp 1 khung hình (giây thứ t) của scene ra PNG để kiểm tra template bằng mắt / test hồi quy."""
    browser_bin = get_browser_path()
    if not browser_bin:
        return None
    out_png = Path(out_png).resolve()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    profile = tempfile.mkdtemp(prefix="mg_profile_")
    page = Path(profile) / "preview.html"
    vendor_js = (_ROOT / "assets" / "vendor" / "tex-svg.js").resolve()
    vendor_url = vendor_js.as_uri() if vendor_js.is_file() else None
    page.write_text(_build_motion_html(scene, duration, width, height, fps, image_path=image_path, vendor_url=vendor_url), encoding="utf-8")
    cmd = _browser_cmd(browser_bin, profile, width, height, f"{page.as_uri()}#preview={t}",
                       extra=[f"--screenshot={out_png}", "--virtual-time-budget=4000"])
    proc = None
    try:
        proc = subprocess.Popen(cmd, **_popen_kwargs())
        proc.wait(timeout=timeout)
    except Exception as e:  # noqa: BLE001
        log.warning("Chụp preview lỗi: %s", e)
    finally:
        _kill_tree(proc)
        shutil.rmtree(profile, ignore_errors=True)
    return out_png if out_png.exists() and out_png.stat().st_size > 1024 else None
