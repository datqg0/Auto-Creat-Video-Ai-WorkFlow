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
    return {"cmp": {"left": {"title": _truncate(left[0], 40), "rows": [_truncate(x, 90) for x in left[1:5]]},
                    "right": {"title": _truncate(right[0], 40), "rows": [_truncate(x, 90) for x in right[1:5]]}}}


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


def _d_story(scene: Scene, lang: str) -> dict | None:
    items = _bullets(scene, 6) or _sentences(scene.narration, 4, 120)
    if len(items) < 2:
        return None
    k = math.ceil(len(items) / 2)
    return {"story": {"problem": [_truncate(x, 100) for x in items[:k][:3]],
                      "solution": [_truncate(x, 100) for x in items[k:][:3]]}}


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
}
_PRIORITY = ["conclusion", "quote_card", "terminal", "scheduler", "bar_benchmark", "github_stats", "algo_trace",
             "neural_flow", "hud", "diff_view", "cpu_scheduler", "git_flow", "attention_map", "comparison",
             "metaphor", "stickman", "architecture"]


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
    elif vt == "chart" and chart:
        add("github_stats" if chart["kind"] == "line" else "bar_benchmark", 20)
    elif vt in ("git", "git_flow"):
        add("git_flow", 20)
    elif vt in ("diff", "diff_view"):
        add("diff_view", 20)
    elif vt in ("comparison", "compare"):
        add("comparison", 20)
    elif vt == "algorithm" and algo == "sorting":
        add("algo_trace", 20)
    elif vt == "algorithm" and algo == "neural_network":
        add("neural_flow", 20)
    elif vt == "challenge":
        add("stickman", 18)
    elif vt == "diagram":
        add("architecture", 18)

    if preset == "bar_chart":
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
        scores["diff_view"] = scores.get("diff_view", 0) + 8
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

const CYAN = '#00f0ff', PINK = '#ff2bd6', GREEN = '#39ff14', YELLOW = '#ffe600', PURPLE = '#a855f7',
      ORANGE = '#ff6b35', BLUE = '#3b82f6', RED = '#ef4444', BG = '#060a14';
const BORDER = 'rgba(56,189,248,0.35)', CARD = 'rgba(12,20,40,0.92)', MUTED = '#94a3b8', DIM = '#64748b';
const PALETTE = [CYAN, PINK, GREEN, YELLOW, PURPLE, ORANGE, BLUE];
const FONT = "'MGFont','Noto Sans','Segoe UI','Be Vietnam Pro',Roboto,'Helvetica Neue',Arial,sans-serif";
const MONO = "'JetBrains Mono','Cascadia Mono',Consolas,'DejaVu Sans Mono','Noto Sans Mono',monospace";

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
  const mx = px(72), top = px(IS_VERTICAL ? 96 : 48), tSize = px(IS_VERTICAL ? 52 : 58);
  const maxW = W - mx * 2 - px(26);
  const tLines = wrapLines(CFG.title, maxW, font(tSize, 'bold'), 2);
  const lh = tSize * 1.2;
  const sSize = px(IS_VERTICAL ? 26 : 28);
  const sLines = CFG.subtitle ? wrapLines(CFG.subtitle, maxW, font(sSize), 2) : [];
  const sLh = sSize * 1.35;
  const bottom = top + tLines.length * lh + (sLines.length ? px(10) + sLines.length * sLh : 0);
  HL = { mx, top, tSize, tLines, lh, sSize, sLines, sLh, bottom };
  return HL;
}
function drawHeader(t) {
  const h = headerLayout();
  const enter = easeOutExpo(clamp(t / 0.7, 0, 1));
  ctx.save(); ctx.globalAlpha = enter; ctx.translate(0, (1 - enter) * -px(24));
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
  const f = font(px(20), 'bold', MONO), tw = measure(L.illustrative, f) + px(28);
  panel(box.x + box.w - tw, box.y - px(34), tw, px(30), { r: 15, fill: 'rgba(255,230,0,0.10)', stroke: YELLOW, lw: 1.5 });
  text(L.illustrative, box.x + box.w - tw / 2, box.y - px(13), f, YELLOW, 'center');
}

// ---------- Người que ----------
function drawStickman(x, groundY, sc, pose, t, color) {
  const headR = 24, spine = 65;
  let tilt = 0, la = 0.5, ra = -0.5, lf = 0.8, rf = -0.8, ll = 0.25, rl = -0.25, jump = 0;
  const bob = Math.sin(t * 5) * 3;
  const oy = groundY - 152 * sc;
  ctx.save(); ctx.translate(x, oy); ctx.scale(sc, sc);
  // bóng đổ
  ctx.save(); ctx.fillStyle = 'rgba(0,240,255,0.16)'; ctx.beginPath(); ctx.ellipse(0, 152, 34, 10, 0, 0, Math.PI * 2); ctx.fill(); ctx.restore();
  if (pose === 'confused') {
    tilt = 0.22 + Math.sin(t * 3) * 0.08; ra = -2.2; rf = -1.4 + Math.sin(t * 8) * 0.28; la = 0.6; lf = -1.1;
    for (let q = 0; q < 3; q++) {
      const p = (t * 0.5 + q * 0.33) % 1, qa = Math.sin(p * Math.PI);
      ctx.save(); ctx.fillStyle = `rgba(255,230,0,${qa})`; ctx.font = font(26, 'bold'); ctx.textAlign = 'center';
      ctx.shadowColor = YELLOW; ctx.shadowBlur = 14; ctx.fillText('?', (q - 1) * 24 + Math.sin(t * 3 + q) * 12, -headR - 22 - p * 55); ctx.restore();
    }
  } else if (pose === 'idea') {
    jump = -Math.abs(Math.sin(t * 6)) * 14; tilt = -0.05; la = -2.3; lf = -0.3; ra = 2.3; rf = 0.3;
    ctx.save(); ctx.translate(0, -headR - 50 + jump + bob);
    const rays = 8, rl2 = 16 + Math.sin(t * 8) * 6;
    ctx.strokeStyle = YELLOW; ctx.lineWidth = 2.5; ctx.shadowColor = YELLOW; ctx.shadowBlur = 18;
    for (let i = 0; i < rays; i++) { const a = i * Math.PI * 2 / rays + t * 0.6; ctx.beginPath(); ctx.moveTo(Math.cos(a) * 20, Math.sin(a) * 20); ctx.lineTo(Math.cos(a) * (20 + rl2), Math.sin(a) * (20 + rl2)); ctx.stroke(); }
    ctx.fillStyle = YELLOW; ctx.beginPath(); ctx.arc(0, 0, 15, 0, 7); ctx.fill();
    ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(0, -2, 6, 0, 7); ctx.fill(); ctx.restore();
  }
  const cy = bob + jump;
  ctx.save(); ctx.translate(0, cy); ctx.rotate(tilt);
  ctx.strokeStyle = color; ctx.lineWidth = 4; ctx.fillStyle = BG; ctx.shadowColor = color; ctx.shadowBlur = 16;
  ctx.beginPath(); ctx.arc(0, 0, headR, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  ctx.fillStyle = YELLOW; ctx.shadowColor = YELLOW; ctx.shadowBlur = 10; ctx.fillRect(-12, -4, 24, 7); ctx.restore();
  const neck = cy + headR, hip = neck + spine, sh = neck + 14;
  ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = 4.5; ctx.lineCap = 'round'; ctx.shadowColor = color; ctx.shadowBlur = 14;
  ctx.beginPath(); ctx.moveTo(0, neck); ctx.lineTo(0, hip); ctx.stroke();
  const limb = (a, f, len1, len2, x0, y0) => { const ex = x0 + Math.sin(a) * len1, ey = y0 + Math.cos(a) * len1;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(ex, ey); ctx.lineTo(ex + Math.sin(a + f) * len2, ey + Math.cos(a + f) * len2); ctx.stroke(); };
  limb(la, lf, 30, 26, 0, sh); limb(ra, rf, 30, 26, 0, sh);
  [ll, rl].forEach(a => { const kx = Math.sin(a) * 34, ky = hip + Math.cos(a) * 34; ctx.beginPath(); ctx.moveTo(0, hip); ctx.lineTo(kx, ky); ctx.lineTo(kx, ky + 30); ctx.stroke(); });
  ctx.restore(); ctx.restore();
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
      const th = tbHeight(s, cw - px(44), fs, { weight: 'bold', maxLines: 5 });
      tb(s, p.x + px(22), p.y + px(48) + Math.max(0, (ch - px(60) - th) / 2), cw - px(44), fs, '#f1f5f9', { weight: 'bold', maxLines: 5 });
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
        const hh = tbHeight(m[1], tw, px(30), { weight: 'bold', maxLines: 1 }), bh = tbHeight(m[2], tw, px(27), { maxLines: 2 });
        const top = y + (ch - hh - bh) / 2;
        tb(m[1], tx, top, tw, px(30), col, { weight: 'bold', maxLines: 1 });
        tb(m[2], tx, top + hh, tw, px(27), '#cbd5e1', { maxLines: 2 });
      } else {
        const fs = px(32), hh = tbHeight(it, tw, fs, { weight: 'bold', maxLines: 3 });
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
    const cnt = Math.min(lines.length, Math.floor(progress(t, 0.5 + k * 0.4, Math.min(DURATION * 0.6, 5)) * (lines.length + 1)));
    lines.slice(0, cnt).forEach((ln, i) => {
      const ly = p.y + hh + px(12) + i * lh;
      ctx.fillStyle = k ? 'rgba(57,255,20,0.08)' : 'rgba(239,68,68,0.08)'; ctx.fillRect(p.x + px(6), ly, p.w - px(12), lh - px(4));
      text(k ? '+' : '-', p.x + px(20), ly + lh * 0.68, f, col);
      text(fitText(ln, p.w - px(80), f), p.x + px(52), ly + lh * 0.68, f, '#e2e8f0');
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
        const re = pop(t, 0.8 + k * 0.3 + i * 0.35, 0.4); if (re <= 0) return;
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
  const labelW = Math.min(px(340), box.w * 0.32), barX = box.x + labelW + px(20), vW = px(210), barMax = box.w - labelW - px(20) - vW;
  const rowH = Math.min(px(100), (box.h - px(110)) / n), barH = rowH * 0.56, y0 = box.y + px(10);
  ctx.save(); ctx.strokeStyle = 'rgba(100,116,139,0.5)'; ctx.lineWidth = px(2); ctx.beginPath(); ctx.moveTo(barX, y0); ctx.lineTo(barX, y0 + n * rowH); ctx.stroke(); ctx.restore();
  b.values.forEach((v, i) => {
    const col = PALETTE[i % PALETTE.length], g = easeOutExpo(progress(t, 0.3 + i * 0.25, 1.3 + i * 0.25)), y = y0 + i * rowH + (rowH - barH) / 2;
    const w = Math.max(px(4), Math.abs(v) / maxV * barMax * g);
    const lh = tbHeight(b.labels[i], labelW, px(26), { weight: 'bold', maxLines: 2 });
    tb(b.labels[i], barX - px(20) - labelW, y + (barH - lh) / 2, labelW, px(26), '#cbd5e1', { weight: 'bold', maxLines: 2, align: 'right' });
    ctx.save(); roundRect(barX, y, w, barH, [0, px(10), px(10), 0]);
    const gr = ctx.createLinearGradient(barX, 0, barX + w, 0); gr.addColorStop(0, col + '55'); gr.addColorStop(1, col);
    ctx.fillStyle = gr; ctx.shadowColor = col; ctx.shadowBlur = px(14); ctx.fill(); ctx.restore();
    text(fmtNum(v * g) + (b.unit ? ' ' + b.unit : ''), barX + w + px(14), y + barH / 2 + px(9), font(px(26), 'bold', MONO), col);
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
  text(fmtNum(cur) + (s.unit ? ' ' + s.unit : ''), box.x + box.w / 2, box.y + px(100), font(px(IS_VERTICAL ? 84 : 96), 'bold', MONO), YELLOW, 'center'); ctx.restore();
  text(s.labels[Math.min(n - 1, Math.round(pos))], box.x + box.w / 2, box.y + px(150), font(px(28), 'bold'), MUTED, 'center');
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

// --- stickman: vấn đề -> giải pháp từ bullets/narration ---
TEMPLATES.stickman = function (t, box) {
  const s = D.story; if (!s) return;
  const tsw = DURATION > 5 ? clamp(DURATION * 0.55, 2.5, DURATION - 2.5) : DURATION * 0.55, resolved = t >= tsw && s.solution.length > 0;
  const col = resolved ? GREEN : CYAN, sc = (IS_VERTICAL ? 1.5 : 1.9) * U;
  let pxL, pyT, pw, ph, gy, fx;
  if (IS_VERTICAL) { fx = box.x + box.w / 2; gy = box.y + px(300); pxL = box.x; pyT = gy + px(40); pw = box.w; ph = box.h - (pyT - box.y); }
  else { fx = box.x + box.w * 0.13; gy = box.y + box.h * 0.8; pxL = box.x + box.w * 0.28; pyT = box.y; pw = box.w * 0.72; ph = box.h; }
  drawStickman(fx, gy, sc, resolved ? 'idea' : 'confused', t, col);
  const items = resolved ? s.solution : s.problem, th = resolved ? GREEN : RED, title = resolved ? L.solution : L.problem;
  const flash = resolved ? 1 - progress(t, tsw, tsw + 0.5) : 0;
  panel(pxL, pyT, pw, ph, { r: 20, fill: resolved ? 'rgba(10,30,20,0.94)' : 'rgba(24,12,18,0.92)', stroke: th, lw: 3, glow: px(20) + flash * px(30) });
  const hh = px(62); roundRect(pxL, pyT, pw, hh, [px(20), px(20), 0, 0]); ctx.fillStyle = resolved ? 'rgba(16,50,28,0.95)' : 'rgba(50,14,20,0.95)'; ctx.fill();
  icon(resolved ? 'bulb' : 'warn', pxL + px(40), pyT + hh / 2, px(30), th); text(title, pxL + px(76), pyT + hh / 2 + px(9), font(px(28), 'bold', MONO), th);
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
    text(k.dur != null ? fmtNum(k.dur) + ' ' + (k.unit || '') : `#${i + 1}`, x + px(14), y + px(76), font(px(20), 'normal', MONO), MUTED);
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
  if (known) text(`${L.total}: ${fmtNum(wsum)} ${tasks[0].unit || ''}`, cx, cpuY + size + px(52), font(px(22), 'normal', MONO), DIM, 'center');
};

// --- git_flow ---
TEMPLATES.git_flow = function (t, box) {
  const cm = D.commits || []; const n = cm.length; if (n < 2) return;
  const x0 = box.x + px(60), x1 = box.x + box.w - px(60), lineY = box.y + box.h * 0.6, brY = lineY - Math.min(px(200), box.h * 0.36);
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
  text(SORT_NAMES[D.algo || 'bubble'], box.x, box.y + px(34), font(px(34), 'bold', MONO), PINK);
  text(`${L.step} ${idx} / ${total - 1}   \u00B7   ${L.compared}: ${st.cmp}   \u00B7   ${L.swaps}: ${st.swp}`, box.x, box.y + px(80), font(px(26), 'bold', MONO), YELLOW);
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
  try { if (document.fonts) { if (CFG.custom_font) { await document.fonts.load('20px MGFont'); await document.fonts.load('bold 20px MGFont'); } await document.fonts.ready; } } catch (e) {}
  if (stockImg && stockImg.decode) { try { await stockImg.decode(); } catch (e) {} }
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
    path = _mg_cfg().get("font_file")
    if not path or not os.path.exists(str(path)):
        return "", False
    try:
        data = base64.b64encode(Path(str(path)).read_bytes()).decode("ascii")
        ext = Path(str(path)).suffix.lower().lstrip(".")
        fmt = {"ttf": "truetype", "otf": "opentype", "woff": "woff", "woff2": "woff2"}.get(ext, "truetype")
        css = (f"@font-face {{ font-family: 'MGFont'; font-weight: 100 900; "
               f"src: url(data:font/{ext};base64,{data}) format('{fmt}'); }}")
        return css, True
    except OSError as e:
        log.debug("Không đọc được font_file: %s", e)
        return "", False


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


def _build_motion_html(
    scene: Scene,
    duration: float,
    width: int,
    height: int,
    fps: int,
    image_path: Path | None = None,
) -> str:
    """Tạo trang HTML Canvas cho 1 scene (template + dữ liệu lấy từ chính scene)."""
    lang = "vi" if str(CONFIG.get("language", "vi")).lower().startswith("vi") else "en"
    img = _read_image_data_url(image_path)
    plan = _plan_scene(scene, bool(img), lang)
    subtitle_src = _sentences(scene.narration, 1, 140)
    css, custom_font = _font_face_css()

    cfg = {
        "width": int(width), "height": int(height), "fps": int(fps), "duration": float(duration),
        "template": plan["template"], "data": plan["data"], "illustrative": plan["illustrative"],
        "title": _clean_title(scene.heading, lang),
        "subtitle": subtitle_src[0] if subtitle_src else "",
        "lang": lang, "labels": _LABELS[lang], "image": img, "custom_font": custom_font,
    }
    html = (_HTML.replace("__LANG__", lang)
            .replace("__FONTFACE__", css)
            .replace("__WIDTH__", str(int(width)))
            .replace("__HEIGHT__", str(int(height)))
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
    html_bytes = _build_motion_html(scene, duration, width, height, fps, image_path=image_path).encode("utf-8")
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
            if urlparse(self.path).path in ("/", "/index.html"):
                self._reply(200, html_bytes, "text/html; charset=utf-8")
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
    page.write_text(_build_motion_html(scene, duration, width, height, fps, image_path=image_path), encoding="utf-8")
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
