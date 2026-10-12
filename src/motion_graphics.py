"""Motion Graphics Engine cao cấp dựng bằng HTML5/Canvas/CSS và Headless Browser (Edge/Chrome).

Phong cách: Tech Studio / Cyberpunk Minimalist (tương tự Fireship, Kurzgesagt, Stripe interactive docs).
- Render trực tiếp bằng Microsoft Edge hoặc Google Chrome/Chromium Headless + HTML5 Canvas 2D + SVG.
- Thu nhận frame tất định (deterministic), không giật lag.
- Đóng gói MP4 chuẩn H.264 qua FFmpeg.
- Tương thích đa nền tảng: Windows, Linux (Ubuntu/CI), Docker, macOS.

TEMPLATES MỚI (v2):
  - "git_flow"       : Git commit graph animation (nhánh, merge, rebase)
  - "attention_map"  : Transformer attention heatmap tương tác
  - "algo_trace"     : Trace thuật toán từng bước với array visualization
  - "github_stats"   : GitHub repo stats, star growth, contributor graph
  - "neural_flow"    : Neural network forward pass với dữ liệu thực
  - "diff_view"      : Code diff animation (thêm/xóa dòng)
  - "matrix_rain"    : Matrix-style animated background + data overlay
  - "architecture"   : Architecture Flow (Data packet animation) - legacy
  - "terminal"       : Terminal / Code typewriter - legacy
  - "comparison"     : Benchmark comparison cards - legacy
  - "hud"            : Tech HUD / Metric Pulse - legacy
"""
from __future__ import annotations

import http.server
import json
import logging
import os
from pathlib import Path
import random
import re
import shutil
import socketserver
import subprocess
import threading
import time
from typing import Any

from .config import CONFIG
from .models import Scene

log = logging.getLogger(__name__)


def get_browser_path() -> str | None:
    """Tìm đường dẫn trình duyệt Headless (Edge hoặc Chrome/Chromium) trên Windows / Linux / macOS."""
    # 1. Các binary phổ biến có trong PATH hệ thống
    candidates_in_path = [
        "msedge",
        "microsoft-edge",
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
        "chrome",
    ]
    for name in candidates_in_path:
        found = shutil.which(name)
        if found:
            return found

    # 2. Các đường dẫn chuẩn trên Windows
    win_candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Edge\Application\msedge.exe"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
    ]
    for p in win_candidates:
        if os.path.exists(p):
            return p

    # 3. Các đường dẫn chuẩn trên Linux / Docker / GitHub Actions
    linux_candidates = [
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        "/usr/bin/microsoft-edge",
        "/snap/bin/chromium",
    ]
    for p in linux_candidates:
        if os.path.exists(p):
            return p

    return None


def get_edge_path() -> str | None:
    """Tương thích ngược cho các module cũ gọi get_edge_path()."""
    return get_browser_path()


def is_motion_graphics_available() -> bool:
    return get_browser_path() is not None


# ---------------------------------------------------------------------------
# Template Detection (smart routing based on scene content)
# ---------------------------------------------------------------------------

def _detect_template(scene: Scene) -> str:
    """Nhận diện template phù hợp nhất dựa trên nội dung scene."""
    title = (scene.heading or "").lower()
    narration = (scene.narration or "")[:200].lower()
    prompt = (scene.visual_prompt or "").lower()
    corpus = f"{title} {narration} {prompt}"

    # GitHub / Git / Open source
    if any(k in corpus for k in ("git ", "github", "commit", "branch", "merge", "rebase", "pull request", "repo", "open source", "clone")):
        return "git_flow"

    # Attention / Transformer / LLM
    if any(k in corpus for k in ("attention", "transformer", "self-attention", "query key value", "head", "embedding", "token", "bert", "gpt", "llm")):
        return "attention_map"

    # Thuật toán trace / sorting / search
    if any(k in corpus for k in ("sorting", "sắp xếp", "search", "tìm kiếm", "mảng", "array", "quicksort", "mergesort", "heap", "bfs", "dfs", "dijkstra", "trace", "từng bước", "step by step")):
        return "algo_trace"

    # GitHub stats / metrics
    if any(k in corpus for k in ("star", "fork", "contributor", "release", "version", "changelog", "repo stats", "downloads", "popularity")):
        return "github_stats"

    # Neural network forward pass
    if any(k in corpus for k in ("neural", "backprop", "gradient", "layer", "activation", "relu", "sigmoid", "mạng neural", "training", "loss", "epoch")):
        return "neural_flow"

    # Code diff / review
    if any(k in corpus for k in ("diff", "thêm dòng", "xóa dòng", "code review", "patch", "refactor", "before after code")):
        return "diff_view"

    # Code / terminal
    if any(k in corpus for k in ("def ", "code", "lệnh", "thuật toán", "cú pháp", "terminal", "log", "api key", "function", "class ", "import")):
        return "terminal"

    # Comparison / benchmark
    if any(k in corpus for k in ("so sánh", "vs", "versus", "khác biệt", "trade-off", "benchmark", "nhanh hơn", "chậm hơn", "hiệu năng")):
        return "comparison"

    # Counter / metrics / HUD
    if any(k in corpus for k in ("triệu", "tỷ", "qps", "req/s", "counter", "con số", "%", "ms", "latency", "throughput")):
        return "hud"

    # Default: architecture flow
    return "architecture"


# ---------------------------------------------------------------------------
# HTML / Canvas Templates
# ---------------------------------------------------------------------------

def _build_motion_html(scene: Scene, duration: float, width: int, height: int, fps: int) -> str:
    """Tạo trang HTML chứa animation canvas tự động khớp với chủ đề và nội dung scene."""
    title = (scene.heading or "PHÂN TÍCH KỸ THUẬT").strip()
    narration_snippet = (scene.narration or "")[:120].strip()
    template_type = _detect_template(scene)

    raw_bullets = scene.bullets or []
    clean_bullets = [str(b).strip() for b in raw_bullets if str(b).strip()]
    bullets_json = json.dumps(clean_bullets, ensure_ascii=False)
    is_vertical = height > width

    # Lấy topic_hint từ visual_prompt để enriching animation
    visual_prompt = (scene.visual_prompt or "").strip()[:300]

    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    width: {width}px;
    height: {height}px;
    background: #060a14;
    overflow: hidden;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
  }}
  #canvas {{
    display: block;
    width: {width}px;
    height: {height}px;
  }}
</style>
</head>
<body>
<canvas id="canvas" width="{width}" height="{height}"></canvas>
<script>
const W = {width};
const H = {height};
const FPS = {fps};
const DURATION = {duration};
const TOTAL_FRAMES = Math.max(1, Math.round(DURATION * FPS));
const TEMPLATE = {json.dumps(template_type)};
const TITLE = {json.dumps(title, ensure_ascii=False)};
const SUBTITLE = {json.dumps(narration_snippet, ensure_ascii=False)};
const BULLETS = {bullets_json};
const IS_VERTICAL = {json.dumps(is_vertical)};
const VISUAL_PROMPT = {json.dumps(visual_prompt, ensure_ascii=False)};

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// ── Bảng màu Premium Cyberpunk Studio ──
const CYAN   = '#00f0ff';
const PINK   = '#ff2bd6';
const GREEN  = '#39ff14';
const YELLOW = '#ffe600';
const PURPLE = '#a855f7';
const ORANGE = '#ff6b35';
const BLUE   = '#3b82f6';
const BG     = '#060a14';
const BG2    = '#0a1020';
const CARD   = 'rgba(12, 20, 40, 0.92)';
const BORDER = 'rgba(56, 189, 248, 0.3)';
const PALETTE = [CYAN, PINK, GREEN, YELLOW, PURPLE, ORANGE, BLUE];

// ── Easing functions ──
function easeOut(x) {{ return 1 - Math.pow(1 - x, 3); }}
function easeInOut(x) {{ return x < 0.5 ? 4*x*x*x : 1-Math.pow(-2*x+2,3)/2; }}
function easeOutExpo(x) {{ return x === 1 ? 1 : 1 - Math.pow(2, -10*x); }}
function lerp(a, b, t) {{ return a + (b-a)*t; }}
function clamp(v, lo, hi) {{ return Math.max(lo, Math.min(hi, v)); }}
function progress(t, start, end) {{ return clamp((t-start)/(end-start), 0, 1); }}

// ── Tiện ích vẽ ──
function roundRect(x, y, w, h, r) {{
  ctx.beginPath();
  ctx.moveTo(x+r, y);
  ctx.arcTo(x+w, y, x+w, y+h, r);
  ctx.arcTo(x+w, y+h, x, y+h, r);
  ctx.arcTo(x, y+h, x, y, r);
  ctx.arcTo(x, y, x+w, y, r);
  ctx.closePath();
}}

function hexagon(cx, cy, r) {{
  ctx.beginPath();
  for (let i=0; i<6; i++) {{
    const a = i*Math.PI/3 - Math.PI/6;
    if (i===0) ctx.moveTo(cx+r*Math.cos(a), cy+r*Math.sin(a));
    else ctx.lineTo(cx+r*Math.cos(a), cy+r*Math.sin(a));
  }}
  ctx.closePath();
}}

function glowText(text, x, y, color, size, blur=18) {{
  ctx.save();
  ctx.shadowColor = color;
  ctx.shadowBlur = blur;
  ctx.fillStyle = color;
  ctx.font = `bold ${{size}}px monospace`;
  ctx.fillText(text, x, y);
  ctx.restore();
}}

function wrapText(text, maxW, fontSize) {{
  const words = text.split(' ');
  const lines = [];
  let line = '';
  ctx.font = `${{fontSize}}px sans-serif`;
  for (const w of words) {{
    const test = line ? line+' '+w : w;
    if (ctx.measureText(test).width > maxW && line) {{
      lines.push(line);
      line = w;
    }} else {{
      line = test;
    }}
  }}
  if (line) lines.push(line);
  return lines;
}}

// ── Nền chung (shared background) ──
function drawBackground(t) {{
  // Nền tối với gradient động
  const grad = ctx.createRadialGradient(W*0.3+Math.sin(t*0.5)*50, H*0.4, 80, W/2, H/2, W*0.85);
  grad.addColorStop(0, 'rgba(14, 165, 233, 0.10)');
  grad.addColorStop(0.5, 'rgba(168, 85, 247, 0.06)');
  grad.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = BG;
  ctx.fillRect(0, 0, W, H);
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, W, H);

  // Dot grid nền
  ctx.fillStyle = 'rgba(56,189,248,0.07)';
  const gs = IS_VERTICAL ? 50 : 65;
  for (let x=gs/2; x<W; x+=gs)
    for (let y=gs/2; y<H; y+=gs) {{
      ctx.beginPath();
      ctx.arc(x, y, 1.2, 0, Math.PI*2);
      ctx.fill();
    }}
}}

// ── Header chung ──
function drawHeader(t) {{
  const enter = easeOutExpo(clamp(t/0.7, 0, 1));
  ctx.save();
  ctx.translate(0, (1-enter)*-25);
  ctx.globalAlpha = enter;

  const hx = IS_VERTICAL ? 48 : 84;
  const hy = IS_VERTICAL ? 120 : 44;

  // Accent bar trái
  ctx.fillStyle = CYAN;
  ctx.shadowColor = CYAN;
  ctx.shadowBlur = 16;
  ctx.fillRect(hx, hy, 7, IS_VERTICAL ? 66 : 90);
  ctx.shadowBlur = 0;

  // Tiêu đề — tăng lên đáng kể
  const titleSize = IS_VERTICAL ? 46 : 58;
  ctx.fillStyle = '#f1f5f9';
  ctx.font = `bold ${{titleSize}}px sans-serif`;
  // Tự wrap nếu title dài
  const lines = wrapText(TITLE, W - hx - 120, titleSize);
  lines.slice(0, 2).forEach((line, li) => {{
    ctx.fillText(line, hx+22, hy + titleSize + li*(titleSize+8));
  }});
  const titleBottom = hy + titleSize + (lines.length > 1 ? (titleSize+8) : 0);

  // Subtitle
  if (SUBTITLE) {{
    ctx.fillStyle = '#94a3b8';
    ctx.font = `${{IS_VERTICAL ? 22 : 28}}px sans-serif`;
    const maxSub = IS_VERTICAL ? 55 : 85;
    const sub = SUBTITLE.length > maxSub ? SUBTITLE.slice(0,maxSub-1)+'…' : SUBTITLE;
    ctx.fillText(sub, hx+22, titleBottom + (IS_VERTICAL ? 34 : 42));
  }}

  ctx.restore();
}}

// ── TEMPLATE: Git Flow ──
function drawGitFlow(t) {{
  const branches = BULLETS.length >= 2 ? BULLETS.slice(0,3) : ['main', 'feature/attention', 'fix/gradient'];
  const n = branches.length;
  const colors = [GREEN, CYAN, YELLOW];

  const baseY = IS_VERTICAL ? H*0.35 : H*0.42;
  const endX = IS_VERTICAL ? W*0.85 : W*0.88;
  const startX = IS_VERTICAL ? W*0.15 : W*0.10;

  // Glyph particles (commits)
  const commits = [0.15, 0.32, 0.52, 0.70, 0.88];
  const branchCommits = [[0.25,0.48,0.65], [0.38,0.58]];

  // Main trunk line
  const trunLen = progress(t, 0.2, 1.0);
  ctx.save();
  ctx.strokeStyle = GREEN;
  ctx.lineWidth = 3;
  ctx.shadowColor = GREEN;
  ctx.shadowBlur = 14;
  ctx.beginPath();
  ctx.moveTo(startX, baseY);
  ctx.lineTo(startX + trunLen*(endX-startX), baseY);
  ctx.stroke();
  ctx.restore();

  // Branch lines
  if (n >= 2) {{
    const brY = IS_VERTICAL ? baseY - 140 : baseY - 120;
    const brStart = startX + (endX-startX)*0.22;
    const brEnd = startX + (endX-startX)*0.72;
    const brLen = progress(t, 0.5, 1.1);
    ctx.save();
    ctx.strokeStyle = CYAN;
    ctx.lineWidth = 2.5;
    ctx.shadowColor = CYAN;
    ctx.shadowBlur = 10;
    ctx.setLineDash([8,4]);
    ctx.beginPath();
    ctx.moveTo(brStart, baseY);
    ctx.bezierCurveTo(brStart+40, baseY, brStart+40, brY, brStart+60, brY);
    ctx.lineTo(brStart + 60 + brLen*(brEnd-brStart-60), brY);
    ctx.stroke();

    if (brLen >= 1.0) {{
      const mLen = progress(t, 1.1, 1.5);
      ctx.beginPath();
      ctx.moveTo(brEnd, brY);
      ctx.bezierCurveTo(brEnd+30, brY, brEnd+30, baseY, brEnd+50*mLen, baseY);
      ctx.stroke();
    }}
    ctx.setLineDash([]);
    ctx.restore();

    // Branch label
    const labelA = easeOutExpo(clamp(progress(t,0.6,1.0),0,1));
    ctx.save();
    ctx.globalAlpha = labelA;
    roundRect(brStart+60, brY-26, 220, 48, 10);
    ctx.fillStyle = 'rgba(0,240,255,0.12)';
    ctx.fill();
    ctx.strokeStyle = CYAN;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = CYAN;
    ctx.font = IS_VERTICAL ? 'bold 18px monospace' : 'bold 22px monospace';
    ctx.fillText('⎇ ' + (branches[1] || 'feature/new'), brStart+74, brY+8);
    ctx.restore();
  }}

  // Commit dots on main
  commits.forEach((pos, ci) => {{
    const cx = startX + pos*(endX-startX);
    const appear = easeOutExpo(clamp(progress(t, pos*0.9, pos*0.9+0.3), 0, 1));
    if (appear <= 0) return;
    ctx.save();
    ctx.globalAlpha = appear;
    ctx.shadowColor = GREEN;
    ctx.shadowBlur = 18;
    ctx.fillStyle = GREEN;
    ctx.beginPath();
    ctx.arc(cx, baseY, 10, 0, Math.PI*2);
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.fillStyle = '#0a1020';
    ctx.font = 'bold 10px monospace';
    ctx.textAlign = 'center';
    ctx.fillText((ci+1).toString(), cx, baseY+4);
    ctx.textAlign = 'left';

    // Hash label
    if (appear > 0.5) {{
      ctx.fillStyle = '#475569';
      ctx.font = IS_VERTICAL ? '16px monospace' : '18px monospace';
      ctx.fillText(`a${{(ci*7+3).toString(16).padStart(5,'0')}}`, cx-20, baseY+40);
    }}
    ctx.restore();
  }});

  // Branch labels (main)
  ctx.save();
  const la = easeOutExpo(clamp(progress(t,0.1,0.6),0,1));
  ctx.globalAlpha = la;
  roundRect(startX-16, baseY+26, 100, 40, 14);
  ctx.fillStyle = 'rgba(57,255,20,0.15)';
  ctx.fill();
  ctx.strokeStyle = GREEN;
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.fillStyle = GREEN;
  ctx.font = IS_VERTICAL ? 'bold 18px monospace' : 'bold 22px monospace';
  ctx.fillText('main', startX-6, baseY+52);
  ctx.restore();

  // HEAD pointer (moving)
  const headPos = startX + Math.min(t*0.7, 0.88)*(endX-startX);
  ctx.save();
  ctx.globalAlpha = clamp(t-0.3, 0, 1);
  ctx.fillStyle = YELLOW;
  ctx.shadowColor = YELLOW;
  ctx.shadowBlur = 24;
  ctx.font = IS_VERTICAL ? 'bold 18px monospace' : 'bold 22px monospace';
  ctx.fillText('▶ HEAD', headPos-24, baseY-32);
  ctx.restore();
}}

// ── TEMPLATE: Attention Map ──
function drawAttentionMap(t) {{
  const tokens = BULLETS.length >= 4
    ? BULLETS.slice(0, 6)
    : ['The', 'cat', 'sat', 'on', 'mat', '.'];
  const n = tokens.length;

  const marginX = IS_VERTICAL ? 60 : 140;
  const gridTop = IS_VERTICAL ? H*0.30 : H*0.26;
  const cellSz = IS_VERTICAL ? Math.min(70, (W-marginX*2)/n) : Math.min(90, (W-marginX*2)/n);
  const gridW = n * cellSz;
  const gridH = n * cellSz;
  const gx0 = (W - gridW) / 2;
  const gy0 = gridTop;

  // Sinh attention weights (mô phỏng: cao trên diagonal + một số off-diagonal)
  const weights = [];
  for (let r=0; r<n; r++) {{
    weights[r] = [];
    for (let c=0; c<n; c++) {{
      // causal + locality pattern
      const d = Math.abs(r-c);
      const v = d===0 ? 0.85 + (r%3)*0.05 : (d===1 ? 0.35+r*0.04 : 0.05+Math.sin(r+c)*0.08);
      weights[r][c] = clamp(v, 0, 1);
    }}
  }}

  const revealT = clamp(progress(t, 0.2, 1.2), 0, 1);

  // Grid cells
  for (let r=0; r<n; r++) {{
    for (let c=0; c<n; c++) {{
      const cellT = clamp(progress(revealT, (r*n+c)/(n*n)*0.6, (r*n+c)/(n*n)*0.6+0.3), 0, 1);
      const w = weights[r][c];
      const x = gx0 + c*cellSz;
      const y = gy0 + r*cellSz;

      // Color: low=dark blue, high=bright cyan/green
      const rr = Math.round(lerp(5, 0, w));
      const gg = Math.round(lerp(10, 240, w));
      const bb = Math.round(lerp(40, 255, w*0.4));
      ctx.fillStyle = `rgba(${{rr}},${{gg}},${{bb}},${{cellT*0.9}})`;
      ctx.fillRect(x+1, y+1, cellSz-2, cellSz-2);

      if (w > 0.5 && cellT > 0.5) {{
        ctx.fillStyle = `rgba(255,255,255,${{cellT*(w-0.4)}})`;
        ctx.font = `bold ${{Math.max(10, cellSz*0.28)}}px monospace`;
        ctx.textAlign = 'center';
        ctx.fillText(w.toFixed(2), x+cellSz/2, y+cellSz/2+5);
        ctx.textAlign = 'left';
      }}
    }}
  }}

  // Token labels (row = query, col = key)
  const labelA = easeOutExpo(clamp(progress(t,0.1,0.5),0,1));
  ctx.globalAlpha = labelA;
  tokens.forEach((tok, i) => {{
    const fz = Math.max(14, cellSz*0.26);
    ctx.fillStyle = '#94a3b8';
    ctx.font = `bold ${{fz}}px monospace`;
    // Column headers (Key)
    ctx.textAlign = 'center';
    ctx.fillText(tok, gx0 + i*cellSz + cellSz/2, gy0-14);
    // Row labels (Query)
    ctx.textAlign = 'right';
    ctx.fillText(tok, gx0-10, gy0 + i*cellSz + cellSz/2+5);
  }});
  ctx.textAlign = 'left';
  ctx.globalAlpha = 1;

  // Scan line highlight (moving row focus)
  const focusRow = Math.floor(t % n);
  ctx.save();
  ctx.strokeStyle = CYAN;
  ctx.lineWidth = 3;
  ctx.shadowColor = CYAN;
  ctx.shadowBlur = 18;
  ctx.setLineDash([5,3]);
  ctx.strokeRect(gx0, gy0 + focusRow*cellSz, gridW, cellSz);
  ctx.restore();

  // Labels
  const lbA = easeOutExpo(clamp(progress(t,0.0,0.4),0,1));
  ctx.save();
  ctx.globalAlpha = lbA*0.9;
  ctx.fillStyle = CYAN;
  ctx.font = `bold ${{IS_VERTICAL?18:22}}px monospace`;
  ctx.textAlign = 'center';
  ctx.fillText('← KEY tokens →', gx0+gridW/2, gy0-38);
  ctx.save();
  ctx.translate(gx0-44, gy0+gridH/2);
  ctx.rotate(-Math.PI/2);
  ctx.fillText('← QUERY tokens →', 0, 0);
  ctx.restore();
  ctx.textAlign = 'left';
  ctx.restore();
}}

// ── TEMPLATE: Algorithm Trace ──
function drawAlgoTrace(t) {{
  const rawData = BULLETS.length >= 4
    ? BULLETS.slice(0,8).map(b => {{ const n=parseInt(b); return isNaN(n)?Math.floor(Math.random()*90+10):n; }})
    : [64, 25, 12, 22, 11, 90, 37, 55];
  const n = rawData.length;
  const barW = IS_VERTICAL ? (W-120)/n : (W-200)/n;
  const maxVal = Math.max(...rawData);
  const maxH = IS_VERTICAL ? H*0.35 : H*0.38;
  const baseY = IS_VERTICAL ? H*0.72 : H*0.78;
  const startX = IS_VERTICAL ? 60 : 100;

  // Bubble sort simulation steps
  const sortSteps = [];
  let arr = [...rawData];
  sortSteps.push([...arr]);
  for (let i=0; i<n; i++) {{
    for (let j=0; j<n-i-1; j++) {{
      if (arr[j] > arr[j+1]) {{
        [arr[j], arr[j+1]] = [arr[j+1], arr[j]];
        sortSteps.push([...arr]);
      }}
    }}
  }}

  const totalSteps = sortSteps.length;
  const stepIdx = Math.min(Math.floor(t*(totalSteps-1)*0.8), totalSteps-1);
  const currentArr = sortSteps[stepIdx];
  const prevArr = stepIdx > 0 ? sortSteps[stepIdx-1] : currentArr;
  const swapPairs = new Set();
  for (let k=0; k<n; k++) {{
    if (currentArr[k] !== prevArr[k]) swapPairs.add(k);
  }}

  // Draw bars
  currentArr.forEach((val, i) => {{
    const bh = (val/maxVal)*maxH;
    const bx = startX + i*(barW+4);
    const by = baseY - bh;
    const isSwap = swapPairs.has(i);
    const isSorted = stepIdx >= totalSteps*0.8 && i >= n - Math.floor(stepIdx/(n+1)) - 1;

    ctx.save();
    ctx.shadowBlur = isSwap ? 24 : 6;

    if (isSwap) {{
      ctx.shadowColor = PINK;
      ctx.fillStyle = PINK;
    }} else if (isSorted) {{
      ctx.shadowColor = GREEN;
      ctx.fillStyle = GREEN;
    }} else {{
      ctx.shadowColor = CYAN;
      const grad = ctx.createLinearGradient(bx, by, bx, baseY);
      grad.addColorStop(0, CYAN);
      grad.addColorStop(1, 'rgba(0,240,255,0.3)');
      ctx.fillStyle = grad;
    }}

    ctx.fillRect(bx, by, barW-2, bh);
    ctx.restore();

    // Value label
    const valA = easeOutExpo(clamp(t*3, 0, 1));
    ctx.save();
    ctx.globalAlpha = valA;
    ctx.fillStyle = '#94a3b8';
    ctx.font = `${{Math.max(10, barW*0.55)}}px monospace`;
    ctx.textAlign = 'center';
    ctx.fillText(val.toString(), bx+barW/2, by-6);
    ctx.textAlign = 'left';
    ctx.restore();
  }});

  // Step counter
  ctx.save();
  ctx.fillStyle = YELLOW;
  ctx.shadowColor = YELLOW;
  ctx.shadowBlur = 10;
  ctx.font = `bold ${{IS_VERTICAL?24:30}}px monospace`;
  ctx.fillText(`Step ${{stepIdx+1}} / ${{totalSteps}}`, startX, IS_VERTICAL ? H*0.24 : H*0.22);
  ctx.shadowBlur = 0;
  ctx.restore();

  // Complexity label
  const compA = easeOutExpo(clamp(progress(t,0.1,0.6),0,1));
  ctx.save();
  ctx.globalAlpha = compA;
  roundRect(IS_VERTICAL?W-190:W-260, IS_VERTICAL?H*0.24-36:H*0.22-36, IS_VERTICAL?172:230, 48, 12);
  ctx.fillStyle = 'rgba(255,43,214,0.12)';
  ctx.fill();
  ctx.strokeStyle = PINK;
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.fillStyle = PINK;
  ctx.font = `bold ${{IS_VERTICAL?18:22}}px monospace`;
  ctx.textAlign = 'center';
  ctx.fillText('O(N²) Bubble Sort', IS_VERTICAL?W-104:W-145, IS_VERTICAL?H*0.24-4:H*0.22-4);
  ctx.textAlign = 'left';
  ctx.restore();
}}

// ── TEMPLATE: GitHub Stats ──
function drawGithubStats(t) {{
  // Star growth chart + contributor hexagons
  const starData = [120, 340, 890, 2100, 4500, 8900, 15000, 28000, 45000, 67000];
  const n = starData.length;
  const maxStars = starData[n-1];

  const chartX = IS_VERTICAL ? 60 : 100;
  const chartY = IS_VERTICAL ? H*0.60 : H*0.68;
  const chartW = IS_VERTICAL ? W-120 : W*0.55;
  const chartH = IS_VERTICAL ? 180 : 200;

  // Axes
  ctx.save();
  ctx.strokeStyle = 'rgba(100,116,139,0.4)';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(chartX, chartY-chartH);
  ctx.lineTo(chartX, chartY);
  ctx.lineTo(chartX+chartW, chartY);
  ctx.stroke();
  ctx.restore();

  // Star growth line (animated draw)
  const drawLen = clamp(progress(t, 0.3, 1.3), 0, 1);
  ctx.save();
  ctx.strokeStyle = YELLOW;
  ctx.lineWidth = 3;
  ctx.shadowColor = YELLOW;
  ctx.shadowBlur = 14;
  ctx.beginPath();
  let started = false;
  for (let i=0; i<n; i++) {{
    const frac = i/(n-1);
    if (frac > drawLen) break;
    const px = chartX + frac*chartW;
    const py = chartY - (starData[i]/maxStars)*chartH;
    if (!started) {{ ctx.moveTo(px, py); started=true; }}
    else ctx.lineTo(px, py);
  }}
  ctx.stroke();

  // Fill under line
  const grad = ctx.createLinearGradient(chartX, chartY-chartH, chartX, chartY);
  grad.addColorStop(0, 'rgba(255,230,0,0.25)');
  grad.addColorStop(1, 'rgba(255,230,0,0)');
  ctx.beginPath();
  ctx.moveTo(chartX, chartY);
  for (let i=0; i<n; i++) {{
    const frac = i/(n-1);
    if (frac > drawLen) break;
    const px = chartX + frac*chartW;
    const py = chartY - (starData[i]/maxStars)*chartH;
    ctx.lineTo(px, py);
  }}
  ctx.lineTo(chartX + Math.min(drawLen,1)*chartW, chartY);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();
  ctx.restore();

  // Star count badge
  const starCount = Math.round(drawLen * maxStars);
  ctx.save();
  ctx.fillStyle = YELLOW;
  ctx.shadowColor = YELLOW;
  ctx.shadowBlur = 28;
  ctx.font = `bold ${{IS_VERTICAL?54:72}}px monospace`;
  ctx.textAlign = 'center';
  ctx.fillText('⭐ ' + (starCount >= 1000 ? (starCount/1000).toFixed(1)+'k' : starCount), W/2, IS_VERTICAL?H*0.27:H*0.26);
  ctx.shadowBlur = 0;
  ctx.fillStyle = '#94a3b8';
  ctx.font = `bold ${{IS_VERTICAL?22:28}}px sans-serif`;
  ctx.fillText('GitHub Stars', W/2, IS_VERTICAL?H*0.27+40:H*0.26+46);
  ctx.textAlign = 'left';
  ctx.restore();

  // Contributor hexagons
  if (!IS_VERTICAL) {{
    const hexColors = [CYAN, GREEN, PINK, YELLOW, PURPLE];
    const hx0 = W*0.65;
    const hy0 = H*0.35;
    for (let i=0; i<5; i++) {{
      const hcx = hx0 + (i%3)*75;
      const hcy = hy0 + Math.floor(i/3)*80;
      const ha = easeOutExpo(clamp(progress(t, 0.2+i*0.1, 0.5+i*0.1), 0, 1));
      ctx.save();
      ctx.globalAlpha = ha;
      ctx.strokeStyle = hexColors[i];
      ctx.lineWidth = 2.5;
      ctx.shadowColor = hexColors[i];
      ctx.shadowBlur = 16;
      hexagon(hcx, hcy, 30);
      ctx.fillStyle = `rgba(0,0,0,0.4)`;
      ctx.fill();
      ctx.stroke();
      ctx.shadowBlur = 0;
      ctx.fillStyle = hexColors[i];
      ctx.font = 'bold 12px monospace';
      ctx.textAlign = 'center';
      ctx.fillText('DEV', hcx, hcy+5);
      ctx.textAlign = 'left';
      ctx.restore();
    }}
  }}
}}

// ── TEMPLATE: Neural Flow ──
function drawNeuralFlow(t) {{
  const rawLayers = BULLETS.length >= 3
    ? BULLETS.slice(0,5).map(b => {{ const n=parseInt(b); return isNaN(n)||n<1?4:Math.min(n,8); }})
    : [4, 6, 6, 4, 2];
  const layers = rawLayers;
  const nL = layers.length;

  const x0 = IS_VERTICAL ? W*0.10 : W*0.08;
  const x1 = IS_VERTICAL ? W*0.90 : W*0.92;
  const y0 = IS_VERTICAL ? H*0.28 : H*0.22;
  const y1 = IS_VERTICAL ? H*0.78 : H*0.84;
  const stepX = (x1-x0)/(nL-1);

  const maxNodes = Math.max(...layers);
  const maxNodeR = Math.min(18, (y1-y0)/(maxNodes*2.5));

  // Draw edges (connections) with activation flow
  const flowT = (t*0.6) % 1.0;
  for (let li=0; li<nL-1; li++) {{
    const lx = x0 + li*stepX;
    const rx = x0 + (li+1)*stepX;
    const ln = layers[li], rn = layers[li+1];
    for (let ni=0; ni<ln; ni++) {{
      const ly = y0 + (ni+0.5)*(y1-y0)/ln;
      for (let rni=0; rni<rn; rni++) {{
        const ry = y0 + (rni+0.5)*(y1-y0)/rn;
        const alpha = 0.06 + 0.04*Math.abs(Math.sin(ni*rni));
        ctx.save();
        ctx.strokeStyle = `rgba(56,189,248,${{alpha}})`;
        ctx.lineWidth = 0.8;
        ctx.beginPath();
        ctx.moveTo(lx, ly);
        ctx.lineTo(rx, ry);
        ctx.stroke();

        // Animated signal pulse
        if (Math.abs((flowT - (li+ni*0.1)/nL)) < 0.12) {{
          const pf = clamp((flowT - (li+ni*0.1)/nL + 0.06)/0.12, 0, 1);
          const px = lerp(lx, rx, pf);
          const py = lerp(ly, ry, pf);
          ctx.fillStyle = `rgba(57,255,20,0.7)`;
          ctx.beginPath();
          ctx.arc(px, py, 3, 0, Math.PI*2);
          ctx.fill();
        }}
        ctx.restore();
      }}
    }}
  }}

  // Draw nodes
  for (let li=0; li<nL; li++) {{
    const lx = x0 + li*stepX;
    const n = layers[li];
    const layerColor = PALETTE[li % PALETTE.length];
    const layerEnter = easeOutExpo(clamp(progress(t, li*0.15, li*0.15+0.4), 0, 1));

    for (let ni=0; ni<n; ni++) {{
      const ny = y0 + (ni+0.5)*(y1-y0)/n;
      const r = maxNodeR * layerEnter;
      const activation = clamp(Math.sin(t*1.5+li+ni)*0.5+0.5, 0, 1);

      ctx.save();
      ctx.shadowColor = layerColor;
      ctx.shadowBlur = 12 * activation;

      // Activation glow
      const nodeGrad = ctx.createRadialGradient(lx, ny, 0, lx, ny, r);
      nodeGrad.addColorStop(0, layerColor);
      nodeGrad.addColorStop(0.6, `rgba(56,189,248,${{activation*0.6}})`);
      nodeGrad.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = nodeGrad;
      ctx.beginPath();
      ctx.arc(lx, ny, r, 0, Math.PI*2);
      ctx.fill();

      ctx.strokeStyle = layerColor;
      ctx.lineWidth = 1.5;
      ctx.globalAlpha = layerEnter;
      ctx.beginPath();
      ctx.arc(lx, ny, r, 0, Math.PI*2);
      ctx.stroke();
      ctx.restore();
    }}

    // Layer label
    const lblA = easeOutExpo(clamp(progress(t, li*0.15+0.2, li*0.15+0.5), 0, 1));
    ctx.save();
    ctx.globalAlpha = lblA;
    ctx.fillStyle = layerColor;
    ctx.font = `bold ${{IS_VERTICAL?18:22}}px monospace`;
    ctx.textAlign = 'center';
    const lbl = li===0 ? 'Input' : li===nL-1 ? 'Output' : `L${{li}}`;
    ctx.fillText(lbl, lx, y1 + (IS_VERTICAL?34:40));
    ctx.fillStyle = '#475569';
    ctx.font = `${{IS_VERTICAL?15:18}}px monospace`;
    ctx.fillText(`${{n}} nodes`, lx, y1 + (IS_VERTICAL?56:64));
    ctx.textAlign = 'left';
    ctx.restore();
  }}
}}

// ── TEMPLATE: Code Diff View ──
function drawDiffView(t) {{
  const oldLines = BULLETS.length >= 2
    ? BULLETS.slice(0, Math.ceil(BULLETS.length/2))
    : ['def attention(Q, K, V):', '    scores = Q @ K.T', '    weights = scores / d_k', '    return weights @ V'];
  const newLines = BULLETS.length >= 4
    ? BULLETS.slice(Math.ceil(BULLETS.length/2))
    : ['def attention(Q, K, V, mask=None):', '    scores = Q @ K.T / sqrt(d_k)', '    if mask: scores += mask', '    weights = softmax(scores)', '    return weights @ V'];

  const panelW = IS_VERTICAL ? W-80 : (W-100)/2 - 10;
  const panelX1 = IS_VERTICAL ? 40 : 50;
  const panelX2 = IS_VERTICAL ? 40 : 50 + panelW + 20;
  const panelY = IS_VERTICAL ? H*0.24 : H*0.24;
  const panelH = IS_VERTICAL ? H*0.27 : H*0.56;
  const lineH = IS_VERTICAL ? 42 : 50;
  const fontSize = IS_VERTICAL ? 20 : 24;

  const drawPanel = (px, py, pw, ph, lines, isNew) => {{
    const color = isNew ? GREEN : '#ef4444';
    ctx.save();
    roundRect(px, py, pw, ph, 14);
    ctx.fillStyle = isNew ? 'rgba(16,50,25,0.9)' : 'rgba(50,10,10,0.9)';
    ctx.fill();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();

    // Header bar
    roundRect(px, py, pw, 50, [14,14,0,0]);
    ctx.fillStyle = isNew ? 'rgba(57,255,20,0.15)' : 'rgba(239,68,68,0.15)';
    ctx.fill();

    const dots = ['#ff5f56','#ffbd2e', isNew?'#27c93f':'#ff5f56'];
    dots.forEach((dc,di) => {{
      ctx.fillStyle = dc;
      ctx.beginPath();
      ctx.arc(px+22+di*26, py+25, 8, 0, Math.PI*2);
      ctx.fill();
    }});

    ctx.fillStyle = '#64748b';
    ctx.font = `bold ${{IS_VERTICAL?16:19}}px monospace`;
    ctx.textAlign = 'center';
    ctx.fillText(isNew ? '+ new version' : '- old version', px+pw/2, py+31);
    ctx.textAlign = 'left';

    const revealCount = Math.min(lines.length, Math.floor(progress(t, 0.2, 1.2) * lines.length * 1.5 + 1));
    lines.slice(0, revealCount).forEach((line, li) => {{
      const ly = py + 60 + li*lineH;
      const sym = isNew ? '+ ' : '- ';
      ctx.fillStyle = isNew ? 'rgba(57,255,20,0.08)' : 'rgba(239,68,68,0.06)';
      ctx.fillRect(px+4, ly-2, pw-8, lineH-2);
      ctx.fillStyle = isNew ? GREEN : '#ef4444';
      ctx.font = `bold ${{fontSize}}px monospace`;
      ctx.fillText(sym, px+14, ly+fontSize*0.75);
      ctx.fillStyle = '#e2e8f0';
      ctx.fillText(line, px+32, ly+fontSize*0.75);
    }});

    ctx.restore();
  }};

  if (IS_VERTICAL) {{
    // Stack vertically
    drawPanel(panelX1, panelY, panelW, panelH, oldLines, false);
    drawPanel(panelX1, panelY + panelH + 20, panelW, panelH, newLines, true);
  }} else {{
    drawPanel(panelX1, panelY, panelW, panelH, oldLines, false);
    drawPanel(panelX2, panelY, panelW, panelH, newLines, true);
  }}
}}

// ── TEMPLATE: Architecture Flow ──
function drawArchitecture(t) {{
  const rawNodes = (BULLETS && BULLETS.length >= 3)
    ? BULLETS.slice(0, 4)
    : ['Client Request', 'Load Balancer', 'API Gateway', 'Service Mesh'];
  const nodes = rawNodes.map(n => n.length > 22 ? n.slice(0,20)+'..' : n);
  const n = nodes.length;

  if (IS_VERTICAL) {{
    const startY = H*0.30, endY = H*0.82;
    const stepY = (endY-startY)/(n-1);
    const cx = W/2;
    const cW = W-120, cH = 105;

    ctx.save();
    ctx.strokeStyle = 'rgba(0,240,255,0.25)';
    ctx.lineWidth = 4;
    ctx.beginPath(); ctx.moveTo(cx, startY); ctx.lineTo(cx, endY); ctx.stroke();
    ctx.strokeStyle = CYAN; ctx.lineWidth = 2;
    ctx.shadowColor = CYAN; ctx.shadowBlur = 18;
    ctx.beginPath(); ctx.moveTo(cx, startY); ctx.lineTo(cx, endY); ctx.stroke();
    ctx.restore();

    nodes.forEach((nd, i) => {{
      const ny = startY + i*stepY;
      const enter = easeOutExpo(clamp(progress(t, i*0.2, i*0.2+0.5), 0, 1));
      ctx.save();
      ctx.translate(cx, ny);
      ctx.scale(enter, enter);
      ctx.globalAlpha = enter;
      roundRect(-cW/2, -cH/2, cW, cH, 18);
      ctx.fillStyle = CARD;
      ctx.fill();
      ctx.strokeStyle = i===1 ? PINK : (i===2 ? CYAN : BORDER);
      ctx.lineWidth = 2.5;
      ctx.shadowColor = i===1 ? PINK : CYAN;
      ctx.shadowBlur = 16;
      ctx.stroke();
      ctx.shadowBlur = 0;
      ctx.fillStyle = i===1 ? PINK : CYAN;
      ctx.font = 'bold 17px monospace';
      ctx.textAlign = 'center';
      ctx.fillText('0'+(i+1), 0, -cH/2+28);
      ctx.fillStyle = '#f1f5f9';
      ctx.font = 'bold 28px sans-serif';
      ctx.fillText(nd, 0, 12);
      ctx.textAlign = 'left';
      ctx.restore();
    }});

    // Animated packet
    const pp = (t*0.7) % 1.0;
    const py = startY + pp*(endY-startY);
    ctx.save();
    ctx.fillStyle = GREEN; ctx.shadowColor = GREEN; ctx.shadowBlur = 22;
    ctx.beginPath(); ctx.arc(cx, py, 9, 0, Math.PI*2); ctx.fill();
    ctx.restore();
    return;
  }}

  // Horizontal layout
  const startX = W*0.10, endX = W*0.90;
  const stepX = (endX-startX)/(n-1);
  const cy = H*0.52;
  const cW = Math.min(280, (endX-startX)/(n-0.3)); // auto-size theo số node
  const cH = 130;

  ctx.save();
  ctx.strokeStyle = 'rgba(0,240,255,0.25)'; ctx.lineWidth = 4;
  ctx.beginPath(); ctx.moveTo(startX, cy); ctx.lineTo(endX, cy); ctx.stroke();
  ctx.strokeStyle = CYAN; ctx.lineWidth = 2;
  ctx.shadowColor = CYAN; ctx.shadowBlur = 18;
  ctx.beginPath(); ctx.moveTo(startX, cy); ctx.lineTo(endX, cy); ctx.stroke();
  ctx.restore();

  nodes.forEach((nd, i) => {{
    const nx = startX + i*stepX;
    const enter = easeOutExpo(clamp(progress(t, i*0.2, i*0.2+0.5), 0, 1));
    ctx.save();
    ctx.translate(nx, cy);
    ctx.scale(enter, enter);
    ctx.globalAlpha = enter;
    roundRect(-cW/2, -cH/2, cW, cH, 18);
    ctx.fillStyle = CARD; ctx.fill();
    ctx.strokeStyle = i===1 ? PINK : (i===2 ? CYAN : BORDER);
    ctx.lineWidth = 2.5;
    ctx.shadowColor = i===1 ? PINK : CYAN; ctx.shadowBlur = 16;
    ctx.stroke(); ctx.shadowBlur = 0;
    ctx.fillStyle = i===1 ? PINK : CYAN;
    ctx.font = 'bold 17px monospace'; ctx.textAlign = 'center';
    ctx.fillText('0'+(i+1), 0, -20);
    ctx.fillStyle = '#f1f5f9';
    ctx.font = `bold ${{Math.max(18, Math.min(26, Math.floor(cW/nd.length*1.2)))}}px sans-serif`;
    ctx.fillText(nd, 0, 26);
    ctx.textAlign = 'left';
    ctx.restore();
  }});

  const pp = (t*0.7) % 1.0;
  const px = startX + pp*(endX-startX);
  ctx.save();
  ctx.fillStyle = GREEN; ctx.shadowColor = GREEN; ctx.shadowBlur = 22;
  ctx.beginPath(); ctx.arc(px, cy, 9, 0, Math.PI*2); ctx.fill();
  for (let tr=1; tr<=3; tr++) {{
    ctx.fillStyle = `rgba(57,255,20,${{0.5/tr}})`;
    ctx.beginPath();
    ctx.arc(px-tr*12, cy, Math.max(2, 9-tr*2), 0, Math.PI*2);
    ctx.fill();
  }}
  ctx.restore();
}}

// ── TEMPLATE: Terminal / Code ──
function drawTerminal(t) {{
  const termW = IS_VERTICAL ? W-80 : W-200;
  const termH = IS_VERTICAL ? Math.round(H*0.52) : H-240;
  const termX = IS_VERTICAL ? 40 : 100;
  const termY = IS_VERTICAL ? Math.round(H*0.25) : Math.round(H*0.20);

  const enter = easeOutExpo(clamp(t/0.6, 0, 1));
  ctx.save();
  ctx.translate(termX, termY);
  ctx.scale(enter, enter);

  // Terminal frame
  roundRect(0, 0, termW, termH, 18);
  ctx.fillStyle = 'rgba(6, 10, 22, 0.97)';
  ctx.fill();
  ctx.strokeStyle = 'rgba(0,240,255,0.40)';
  ctx.lineWidth = 2;
  ctx.shadowColor = 'rgba(0,240,255,0.18)'; ctx.shadowBlur = 24;
  ctx.stroke(); ctx.shadowBlur = 0;

  // Header bar
  roundRect(0, 0, termW, 54, [18,18,0,0]);
  ctx.fillStyle = 'rgba(20,30,50,0.85)'; ctx.fill();

  ['#ff5f56','#ffbd2e','#27c93f'].forEach((c,idx) => {{
    ctx.fillStyle = c;
    ctx.beginPath();
    ctx.arc(26+idx*28, 27, 9, 0, Math.PI*2);
    ctx.fill();
  }});
  ctx.fillStyle = '#64748b';
  ctx.font = IS_VERTICAL ? 'bold 16px monospace' : 'bold 19px monospace';
  ctx.textAlign = 'center';
  ctx.fillText('algorithm-trace.py  ●', termW/2, 34);
  ctx.textAlign = 'left';

  const rawLines = (BULLETS && BULLETS.length >= 2) ? BULLETS : [
    '# Pseudocode: Attention Mechanism',
    'def attention(Q, K, V):',
    '    d_k = Q.shape[-1]',
    '    score = Q @ K.T / sqrt(d_k)',
    '    weights = softmax(score)',
    '    return weights @ V',
  ];
  const maxC = IS_VERTICAL ? 38 : 58;
  const lines = rawLines.map(l => l.length > maxC ? l.slice(0,maxC-2)+'..' : l);
  const revealed = Math.min(lines.length, Math.floor(clamp(t-0.4,0,10)*3.2)+1);
  const lhStep = IS_VERTICAL ? 44 : 52;

  ctx.font = IS_VERTICAL ? 'bold 22px monospace' : 'bold 27px monospace';
  for (let i=0; i<revealed; i++) {{
    const ly = 96 + i*lhStep;
    ctx.fillStyle = '#374151';
    ctx.fillText((i+1).toString().padStart(2,' '), 26, ly);
    const line = lines[i];
    if (line.startsWith('def ') || line.startsWith('class ')) ctx.fillStyle = PINK;
    else if (line.trim().startsWith('#')) ctx.fillStyle = '#4b5563';
    else if (line.includes('return')) ctx.fillStyle = YELLOW;
    else if (line.includes('=') || line.includes(':')) ctx.fillStyle = CYAN;
    else ctx.fillStyle = '#94a3b8';
    ctx.fillText(line, 80, ly);
  }}
  // Blinking cursor
  if (Math.floor(t*2)%2===0) {{
    const cy2 = 96 + Math.max(0, revealed-1)*lhStep;
    ctx.fillStyle = CYAN;
    ctx.fillRect(termW-60, cy2-20, 13, 28);
  }}
  ctx.restore();
}}

// ── TEMPLATE: Comparison ──
function drawComparison(t) {{
  const enter = easeOutExpo(clamp(t/0.7, 0, 1));
  const items = [
    {{ label: BULLETS[0] || 'Cách truyền thống', items: ['Độ phức tạp O(N²)', 'Tắc nghẽn pipeline', 'Tốn 12 chu kỳ'], color: '#ef4444', ok: false }},
    {{ label: BULLETS[1] || 'Kiến trúc tối ưu',  items: ['Tối ưu O(N log N)', 'Forwarding path',  'Chỉ 3 chu kỳ'], color: GREEN, ok: true  }},
  ];

  const pW = IS_VERTICAL ? W-100 : (W-180)/2;
  const pH = IS_VERTICAL ? H*0.28 : H*0.48;
  const pY = IS_VERTICAL ? H*0.29 : H*0.26;
  const pGap = IS_VERTICAL ? H*0.30 : 80;
  const positions = IS_VERTICAL
    ? [[50, pY], [50, pY+pGap]]
    : [[90, pY], [90+pW+80, pY]];

  items.forEach((item, pi) => {{
    const label = item.label;
    const rows = item.items;
    const color = item.color;
    const ok = item.ok;
    const [px, py] = positions[pi];
    ctx.save();
    ctx.translate(px, py); ctx.scale(enter, enter); ctx.globalAlpha = enter;
    roundRect(0, 0, pW, pH, 20);
    ctx.fillStyle = ok ? 'rgba(16,50,25,0.9)' : 'rgba(50,10,10,0.9)';
    ctx.fill();
    ctx.strokeStyle = color; ctx.lineWidth = 2.5;
    ctx.shadowColor = color; ctx.shadowBlur = ok ? 20 : 0;
    ctx.stroke(); ctx.shadowBlur = 0;

    ctx.fillStyle = color;
    ctx.font = `bold ${{IS_VERTICAL?26:32}}px sans-serif`;
    ctx.fillText((ok?'✓ ':'✗ ') + label, 28, 56);

    rows.forEach((row, ri) => {{
      ctx.fillStyle = ok ? '#6ee7b7' : '#fca5a5';
      ctx.font = `${{IS_VERTICAL?21:26}}px sans-serif`;
      ctx.fillText((ok?'→ ':'× ') + row, 28, 108 + ri*(IS_VERTICAL?44:54));
    }});

    if (ok) {{
      const ba = easeOutExpo(clamp(progress(t,1.0,1.5),0,1));
      ctx.save();
      ctx.translate(pW/2, pH-46); ctx.scale(ba, ba);
      roundRect(-110,-24,220,48,24);
      ctx.fillStyle = YELLOW; ctx.shadowColor = YELLOW; ctx.shadowBlur = 22;
      ctx.fill(); ctx.shadowBlur = 0;
      ctx.fillStyle = '#000'; ctx.font = 'bold 26px sans-serif'; ctx.textAlign='center';
      ctx.fillText('TỐC ĐỘ GẤP 4X', 0, 10);
      ctx.textAlign='left'; ctx.restore();
    }}
    ctx.restore();
  }});
}}

// ── TEMPLATE: HUD Metrics ──
function drawHUD(t) {{
  const cx = W/2;
  const cy = IS_VERTICAL ? H*0.5 : H*0.54;
  const r  = IS_VERTICAL ? 140 : 180;

  // Rotating outer ring
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(t*1.1);
  ctx.strokeStyle = 'rgba(0,240,255,0.15)'; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.arc(0, 0, r, 0, Math.PI*2); ctx.stroke();
  [CYAN, PINK].forEach((c, ci) => {{
    ctx.strokeStyle = c; ctx.lineWidth = 4;
    ctx.shadowColor = c; ctx.shadowBlur = 12;
    ctx.beginPath();
    ctx.arc(0, 0, r+22, ci*Math.PI, ci*Math.PI+Math.PI*0.65);
    ctx.stroke();
  }});
  ctx.restore();

  // Counter
  const p = Math.min(t/(DURATION*0.75), 1.0);
  const val = Math.round(p * 1000000);
  ctx.save();
  ctx.textAlign = 'center';
  ctx.fillStyle = '#f8fafc';
  ctx.font = IS_VERTICAL ? 'bold 68px monospace' : 'bold 88px monospace';
  ctx.shadowColor = CYAN; ctx.shadowBlur = 28;
  ctx.fillText(val.toLocaleString()+'+', cx, cy+22);
  ctx.shadowBlur = 0;
  ctx.fillStyle = '#94a3b8';
  ctx.font = IS_VERTICAL ? 'bold 22px sans-serif' : 'bold 28px sans-serif';
  ctx.fillText('OPS / SECOND', cx, cy + (IS_VERTICAL?72:88));

  // Side metrics
  const metrics = [
    {{ label: 'Latency', val: (1.2-t*0.008).toFixed(2)+'ms', c: GREEN }},
    {{ label: 'Cache HIT', val: Math.round(95+t*2)+'%', c: YELLOW }},
    {{ label: 'Nodes', val: Math.round(t*15+4).toString(), c: CYAN }},
  ];
  const mA = easeOutExpo(clamp(progress(t,0.3,0.9),0,1));
  ctx.globalAlpha = mA;
  metrics.forEach((m, mi) => {{
    const mx = IS_VERTICAL ? cx + (mi-1)*145 : cx + (mi-1)*230;
    const my = IS_VERTICAL ? cy + 140 : cy + 165;
    roundRect(mx-68, my-30, 136, 66, 14);
    ctx.fillStyle = 'rgba(12,20,40,0.9)'; ctx.fill();
    ctx.strokeStyle = m.c; ctx.lineWidth = 2; ctx.stroke();
    ctx.fillStyle = m.c; ctx.font = `bold ${{IS_VERTICAL?20:26}}px monospace`;
    ctx.fillText(m.val, mx, my+8);
    ctx.fillStyle = '#64748b'; ctx.font = `bold ${{IS_VERTICAL?15:18}}px sans-serif`;
    ctx.fillText(m.label, mx-60+8, my+36);
  }});
  ctx.textAlign = 'left';
  ctx.restore();
}}

// ── Master Render ──
function renderFrameAt(f) {{
  const t = f / FPS;
  drawBackground(t);
  drawHeader(t);
  switch (TEMPLATE) {{
    case 'git_flow':      drawGitFlow(t); break;
    case 'attention_map': drawAttentionMap(t); break;
    case 'algo_trace':    drawAlgoTrace(t); break;
    case 'github_stats':  drawGithubStats(t); break;
    case 'neural_flow':   drawNeuralFlow(t); break;
    case 'diff_view':     drawDiffView(t); break;
    case 'terminal':      drawTerminal(t); break;
    case 'comparison':    drawComparison(t); break;
    case 'hud':           drawHUD(t); break;
    default:              drawArchitecture(t); break;
  }}
}}

// ── Frame Extraction Loop ──
async function runCapture() {{
  for (let f = 0; f < TOTAL_FRAMES; f++) {{
    renderFrameAt(f);
    const blob = await new Promise(res => canvas.toBlob(res, 'image/jpeg', 0.92));
    const buf = await blob.arrayBuffer();
    await fetch('/frame?idx=' + f, {{ method: 'POST', body: buf }});
  }}
  await fetch('/done', {{ method: 'POST' }});
}}

runCapture();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Runner Engine
# ---------------------------------------------------------------------------

def render_motion_graphic_scene(
    scene: Scene,
    out_dir: Path,
    duration: float,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    timeout: int = 120,
) -> Path | None:
    """Render 1 scene thành video Motion Graphic bằng Edge/Chrome Headless + HTML5 Canvas.

    Trả về Path tới file out.mp4 nếu thành công, None nếu thất bại.
    """
    browser_bin = get_browser_path()
    if not browser_bin:
        log.warning("Không tìm thấy Microsoft Edge hoặc Google Chrome để render Motion Graphics.")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Log detected template for debugging
    template = _detect_template(scene)
    log.info("Motion Graphics template: '%s' cho scene: %s", template, (scene.heading or "")[:40])

    # 1. Khởi tạo server HTTP local nhận frame (cổng ngẫu nhiên do OS cấp)
    captured_frames: list[int] = []
    is_done = threading.Event()
    total_expected = max(1, round(duration * fps))

    html_content = _build_motion_html(scene, duration, width, height, fps)

    class FrameHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if "/done" in self.path:
                is_done.set()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html_content.encode("utf-8"))

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            data = self.rfile.read(length)

            if "/frame" in self.path:
                idx = len(captured_frames)
                m = re.search(r"idx=(\d+)", self.path)
                if m:
                    idx = int(m.group(1))
                frame_file = frames_dir / f"frame_{idx:05d}.jpg"
                frame_file.write_bytes(data)
                captured_frames.append(idx)
                if len(captured_frames) >= total_expected:
                    is_done.set()

            elif "/done" in self.path:
                is_done.set()

            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, format, *args):
            pass

    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    server = ReusableTCPServer(("127.0.0.1", 0), FrameHandler)
    port = server.server_address[1]
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    # 2. Khởi chạy Edge/Chrome Headless mở trang render
    cmd = [
        browser_bin,
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--disable-dev-shm-usage",
        f"--window-size={width},{height}",
        "--hide-scrollbars",
        f"http://127.0.0.1:{port}",
    ]

    t0 = time.time()
    proc = None
    try:
        proc = subprocess.Popen(cmd)
        # Chờ trình duyệt render xong và gửi tín hiệu /done
        done = is_done.wait(timeout=timeout)
        if not done:
            log.warning("Motion Graphics render vượt timeout %ds.", timeout)
    except Exception as e:
        log.warning("Khởi chạy trình duyệt headless lỗi: %s", e)
    finally:
        if proc:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=1.0)
        except Exception:
            pass

    dt = time.time() - t0
    total_expected = max(1, round(duration * fps))
    log.info(
        "Motion Graphics [%s]: Đã thu được %d/%d frames trong %.2fs",
        template, len(captured_frames), total_expected, dt,
    )

    if not captured_frames or len(captured_frames) < max(3, total_expected * 0.6):
        log.warning("Số lượng frame thu được quá ít, Motion Graphics thất bại.")
        return None

    # 3. Dùng FFmpeg ghép frame thành video MP4 chuẩn
    out_mp4 = out_dir / "out.mp4"
    try:
        import imageio_ffmpeg
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg_bin = "ffmpeg"

    ff_cmd = [
        ffmpeg_bin,
        "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%05d.jpg"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_mp4),
    ]

    try:
        res = subprocess.run(ff_cmd, capture_output=True, text=True, timeout=60)
        if res.returncode == 0 and out_mp4.exists() and out_mp4.stat().st_size > 1024:
            log.info("Motion Graphics render thành công: %s (%.1fMB)", out_mp4.name, out_mp4.stat().st_size / 1e6)
            # Dọn dẹp frames
            try:
                for f in frames_dir.glob("*.jpg"):
                    f.unlink()
                frames_dir.rmdir()
            except Exception:
                pass
            return out_mp4
        else:
            log.warning("FFmpeg ghép MP4 lỗi: %s", res.stderr[-400:])
    except Exception as e:
        log.warning("Lỗi khi chạy FFmpeg ghép video: %s", e)

    return None
