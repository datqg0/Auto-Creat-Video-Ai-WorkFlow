"""Motion Graphics Engine cao cấp dựng bằng HTML5/Canvas/CSS và Headless Browser (Edge/Chrome).

Phong cách: Tech Studio / Cyberpunk Minimalist (tương tự Fireship, Kurzgesagt, Stripe interactive docs).
- Render trực tiếp bằng Microsoft Edge hoặc Google Chrome/Chromium Headless + HTML5 Canvas 2D + SVG.
- Thu nhận frame tất định (deterministic), không giật lag.
- Đóng gói MP4 chuẩn H.264 qua FFmpeg.
- Tương thích đa nền tảng: Windows, Linux (Ubuntu/CI), Docker, macOS.
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
# HTML / Canvas Templates
# ---------------------------------------------------------------------------

def _build_motion_html(scene: Scene, duration: float, width: int, height: int, fps: int) -> str:
    """Tạo trang HTML chứa animation canvas tự động khớp với chủ đề và nội dung scene."""
    title = (scene.heading or "KIẾN TRÚC HỆ THỐNG").strip()
    narration_snippet = (scene.narration or "")[:120].strip()
    prompt = (scene.visual_prompt or "").lower()
    text_corpus = f"{title} {narration_snippet} {prompt}".lower()

    # Nhận diện loại template phù hợp nhất
    if any(k in text_corpus for k in ("so sánh", "vs", "versus", "khác biệt", "trade-off", "benchmark", "nhanh hơn")):
        template_type = "comparison"
    elif any(k in text_corpus for k in ("def ", "code", "lệnh", "thuật toán", "cú pháp", "terminal", "log", "api key")):
        template_type = "terminal"
    elif any(k in text_corpus for k in ("triệu", "tỷ", "qps", "req/s", "counter", "con số", "%", "hiệu năng", "ms")):
        template_type = "hud"
    else:
        template_type = "architecture"

    raw_bullets = scene.bullets or []
    clean_bullets = [str(b).strip() for b in raw_bullets if str(b).strip()]
    bullets_json = json.dumps(clean_bullets, ensure_ascii=False)
    is_vertical = height > width

    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    width: {width}px;
    height: {height}px;
    background: #080b12;
    overflow: hidden;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
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

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// Bảng màu Cyberpunk Studio
const CY = '#00f0ff';
const PK = '#ff2bd6';
const GR = '#39ff14';
const YE = '#ffe600';
const PU = '#a855f7';
const BG = '#080b12';
const CARD_BG = 'rgba(15, 23, 42, 0.85)';
const BORDER = 'rgba(56, 189, 248, 0.35)';

function easeOutExpo(x) {{
  return x === 1 ? 1 : 1 - Math.pow(2, -10 * x);
}}

function easeInOutQuad(x) {{
  return x < 0.5 ? 2 * x * x : 1 - Math.pow(-2 * x + 2, 2) / 2;
}}

function drawBackground(t) {{
  ctx.fillStyle = BG;
  ctx.fillRect(0, 0, W, H);

  // Radial ambient gradient
  const grad = ctx.createRadialGradient(W / 2, H / 2, 50, W / 2, H / 2, W * 0.7);
  grad.addColorStop(0, 'rgba(14, 165, 233, 0.12)');
  grad.addColorStop(0.6, 'rgba(168, 85, 247, 0.05)');
  grad.addColorStop(1, 'rgba(0, 0, 0, 0)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, W, H);

  // Subtle grid
  ctx.strokeStyle = 'rgba(30, 41, 59, 0.4)';
  ctx.lineWidth = 1;
  const gridSize = 60;
  for (let x = 0; x < W; x += gridSize) {{
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, H);
    ctx.stroke();
  }}
  for (let y = 0; y < H; y += gridSize) {{
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(W, y);
    ctx.stroke();
  }}
}}

function drawHeader(t) {{
  const enter = easeOutExpo(Math.min(t / 0.8, 1.0));
  ctx.save();
  ctx.translate(0, (1 - enter) * -30);
  ctx.globalAlpha = enter;

  // Title badge
  ctx.fillStyle = 'rgba(0, 240, 255, 0.12)';
  ctx.strokeStyle = 'rgba(0, 240, 255, 0.4)';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  const badgeX = IS_VERTICAL ? 40 : 80;
  const badgeY = IS_VERTICAL ? 60 : 50;
  ctx.roundRect(badgeX, badgeY, 180, 36, 18);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = CY;
  ctx.font = 'bold 15px monospace';
  ctx.fillText('CORE MECHANISM', badgeX + 25, badgeY + 23);

  // Main Heading
  ctx.fillStyle = '#f8fafc';
  ctx.font = IS_VERTICAL ? 'bold 34px sans-serif' : 'bold 42px sans-serif';
  ctx.shadowColor = 'rgba(0, 240, 255, 0.5)';
  ctx.shadowBlur = 15;
  const headerX = IS_VERTICAL ? 40 : 80;
  const headerY = IS_VERTICAL ? badgeY + 80 : 130;
  const displayTitle = TITLE.length > 50 ? TITLE.slice(0, 48) + '...' : TITLE;
  ctx.fillText(displayTitle, headerX, headerY);
  ctx.shadowBlur = 0;

  // Subtitle
  if (SUBTITLE) {{
    ctx.fillStyle = '#94a3b8';
    ctx.font = IS_VERTICAL ? '18px sans-serif' : '22px sans-serif';
    const subY = headerY + (IS_VERTICAL ? 35 : 40);
    const maxSubLen = IS_VERTICAL ? 55 : 85;
    const displaySub = SUBTITLE.length > maxSubLen ? SUBTITLE.slice(0, maxSubLen - 2) + '...' : SUBTITLE;
    ctx.fillText(displaySub, headerX, subY);
  }}
  ctx.restore();
}}

// -------------------------------------------------------------
// Template: Architecture Flow (Data packet animation)
// -------------------------------------------------------------
function drawArchitecture(t) {{
  const rawNodes = (BULLETS && BULLETS.length >= 3)
    ? BULLETS.slice(0, 4)
    : ['Client (App/Web)', 'API Gateway', 'Processing Engine', 'Distributed Cache'];

  const nodes = rawNodes.map(n => n.length > 20 ? n.slice(0, 18) + '..' : n);
  const n = nodes.length;

  if (IS_VERTICAL) {{
    const startY = 320;
    const endY = H - 220;
    const stepY = (endY - startY) / (n - 1);
    const centerX = W / 2;
    const cardW = W - 160;
    const cardH = 95;

    ctx.save();
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.3)';
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(centerX, startY);
    ctx.lineTo(centerX, endY);
    ctx.stroke();

    ctx.strokeStyle = CY;
    ctx.lineWidth = 2;
    ctx.shadowColor = CY;
    ctx.shadowBlur = 18;
    ctx.beginPath();
    ctx.moveTo(centerX, startY);
    ctx.lineTo(centerX, endY);
    ctx.stroke();
    ctx.restore();

    for (let i = 0; i < n; i++) {{
      const x = centerX;
      const y = startY + i * stepY;
      const nodeEnter = easeOutExpo(Math.min(Math.max((t - i * 0.25) / 0.6, 0.0), 1.0));

      ctx.save();
      ctx.translate(x, y);
      ctx.scale(nodeEnter, nodeEnter);
      ctx.globalAlpha = nodeEnter;

      ctx.fillStyle = CARD_BG;
      ctx.strokeStyle = i === 1 ? PK : (i === 2 ? CY : BORDER);
      ctx.lineWidth = 2;
      ctx.shadowColor = i === 1 ? PK : CY;
      ctx.shadowBlur = 14;
      ctx.beginPath();
      ctx.roundRect(-cardW / 2, -cardH / 2, cardW, cardH, 16);
      ctx.fill();
      ctx.stroke();
      ctx.shadowBlur = 0;

      ctx.fillStyle = i === 1 ? 'rgba(255, 43, 214, 0.2)' : 'rgba(0, 240, 255, 0.15)';
      ctx.beginPath();
      ctx.arc(-cardW / 2 + 50, 0, 22, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = i === 1 ? PK : CY;
      ctx.font = 'bold 16px monospace';
      ctx.textAlign = 'center';
      ctx.fillText('0' + (i + 1), -cardW / 2 + 50, 6);

      ctx.fillStyle = '#f1f5f9';
      ctx.font = 'bold 22px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(nodes[i], -cardW / 2 + 90, 8);

      ctx.restore();
    }}

    const packetProgress = (t * 0.8) % 1.0;
    const currentPos = startY + packetProgress * (endY - startY);
    ctx.save();
    ctx.fillStyle = GR;
    ctx.shadowColor = GR;
    ctx.shadowBlur = 24;
    ctx.beginPath();
    ctx.arc(centerX, currentPos, 10, 0, Math.PI * 2);
    ctx.fill();
    for (let tr = 1; tr <= 4; tr++) {{
      ctx.fillStyle = 'rgba(57, 255, 20, ' + (0.6 / tr) + ')';
      ctx.beginPath();
      ctx.arc(centerX, currentPos - tr * 14, Math.max(2, 10 - tr * 2), 0, Math.PI * 2);
      ctx.fill();
    }}
    ctx.restore();
    return;
  }}

  const startX = 140;
  const endX = W - 140;
  const stepX = (endX - startX) / (n - 1);
  const centerY = H / 2 + 50;
  const cardW = 200;
  const cardH = 120;

  ctx.save();
  ctx.strokeStyle = 'rgba(56, 189, 248, 0.3)';
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.moveTo(startX, centerY);
  ctx.lineTo(endX, centerY);
  ctx.stroke();

  ctx.strokeStyle = CY;
  ctx.lineWidth = 2;
  ctx.shadowColor = CY;
  ctx.shadowBlur = 18;
  ctx.beginPath();
  ctx.moveTo(startX, centerY);
  ctx.lineTo(endX, centerY);
  ctx.stroke();
  ctx.restore();

  for (let i = 0; i < n; i++) {{
    const x = startX + i * stepX;
    const y = centerY;
    const nodeEnter = easeOutExpo(Math.min(Math.max((t - i * 0.25) / 0.6, 0.0), 1.0));

    ctx.save();
    ctx.translate(x, y);
    ctx.scale(nodeEnter, nodeEnter);
    ctx.globalAlpha = nodeEnter;

    ctx.fillStyle = CARD_BG;
    ctx.strokeStyle = i === 1 ? PK : (i === 2 ? CY : BORDER);
    ctx.lineWidth = 2;
    ctx.shadowColor = i === 1 ? PK : CY;
    ctx.shadowBlur = 14;
    ctx.beginPath();
    ctx.roundRect(-cardW / 2, -cardH / 2, cardW, cardH, 16);
    ctx.fill();
    ctx.stroke();
    ctx.shadowBlur = 0;

    ctx.fillStyle = i === 1 ? 'rgba(255, 43, 214, 0.2)' : 'rgba(0, 240, 255, 0.15)';
    ctx.beginPath();
    ctx.arc(0, -18, 22, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = i === 1 ? PK : CY;
    ctx.font = 'bold 16px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('0' + (i + 1), 0, -12);

    ctx.fillStyle = '#f1f5f9';
    ctx.font = 'bold 18px sans-serif';
    ctx.fillText(nodes[i], 0, 32);

    ctx.restore();
  }}

  const packetProgress = (t * 0.8) % 1.0;
  const currentPos = startX + packetProgress * (endX - startX);

  ctx.save();
  ctx.fillStyle = GR;
  ctx.shadowColor = GR;
  ctx.shadowBlur = 24;
  ctx.beginPath();
  ctx.arc(currentPos, centerY, 10, 0, Math.PI * 2);
  ctx.fill();

  for (let tr = 1; tr <= 4; tr++) {{
    ctx.fillStyle = 'rgba(57, 255, 20, ' + (0.6 / tr) + ')';
    ctx.beginPath();
    ctx.arc(currentPos - tr * 14, centerY, Math.max(2, 10 - tr * 2), 0, Math.PI * 2);
    ctx.fill();
  }}
  ctx.restore();
}}

// -------------------------------------------------------------
// Template: Terminal / Code
// -------------------------------------------------------------
function drawTerminal(t) {{
  const termW = IS_VERTICAL ? W - 80 : W - 240;
  const termH = IS_VERTICAL ? H - 380 : H - 270;
  const termX = IS_VERTICAL ? 40 : 120;
  const termY = IS_VERTICAL ? 260 : 220;

  const enter = easeOutExpo(Math.min(t / 0.6, 1.0));
  ctx.save();
  ctx.translate(termX, termY);
  ctx.scale(enter, enter);

  ctx.fillStyle = 'rgba(10, 15, 29, 0.95)';
  ctx.strokeStyle = 'rgba(0, 240, 255, 0.4)';
  ctx.lineWidth = 2;
  ctx.shadowColor = 'rgba(0, 240, 255, 0.2)';
  ctx.shadowBlur = 25;
  ctx.beginPath();
  ctx.roundRect(0, 0, termW, termH, 16);
  ctx.fill();
  ctx.stroke();
  ctx.shadowBlur = 0;

  ctx.fillStyle = 'rgba(30, 41, 59, 0.7)';
  ctx.beginPath();
  ctx.roundRect(0, 0, termW, 46, [16, 16, 0, 0]);
  ctx.fill();

  const dots = ['#ff5f56', '#ffbd2e', '#27c93f'];
  dots.forEach((c, idx) => {{
    ctx.fillStyle = c;
    ctx.beginPath();
    ctx.arc(26 + idx * 22, 23, 7, 0, Math.PI * 2);
    ctx.fill();
  }});

  ctx.fillStyle = '#94a3b8';
  ctx.font = '15px monospace';
  ctx.fillText('bash - algorithm-engine.py', termW / 2 - 80, 28);

  const rawLines = (BULLETS && BULLETS.length >= 2)
    ? BULLETS
    : [
      'def execute_pipeline(stream_context):',
      '    # 1. Parse and validate stream buffers',
      '    buffer = MemoryAlloc.acquire(capacity=1024)',
      '    event = Dispatcher.route(stream_context)',
      '    if event.is_valid():',
      '        return PipelineResult.SUCCESS(speedup="10x")',
    ];

  const maxChars = IS_VERTICAL ? 42 : 65;
  const lines = rawLines.map(l => l.length > maxChars ? l.slice(0, maxChars - 2) + '..' : l);

  const revealedCount = Math.min(lines.length, Math.floor((t - 0.5) * 3.5) + 1);
  ctx.font = IS_VERTICAL ? '18px monospace' : '22px monospace';
  for (let i = 0; i < revealedCount; i++) {{
    const lineY = 90 + i * (IS_VERTICAL ? 36 : 42);

    ctx.fillStyle = '#475569';
    ctx.fillText((i + 1).toString().padStart(2, ' '), 30, lineY);

    const text = lines[i];
    if (text.startsWith('def ') || text.startsWith('class ')) {{
      ctx.fillStyle = PK;
    }} else if (text.includes('#')) {{
      ctx.fillStyle = '#64748b';
    }} else if (text.includes('return ') || text.includes('if ')) {{
      ctx.fillStyle = YE;
    }} else {{
      ctx.fillStyle = '#38bdf8';
    }}
    ctx.fillText(text, 75, lineY);
  }}

  if (Math.floor(t * 3) % 2 === 0) {{
    const cursorY = 90 + Math.max(0, revealedCount - 1) * (IS_VERTICAL ? 36 : 42);
    ctx.fillStyle = CY;
    ctx.fillRect(termW - 80, cursorY - 18, 12, 24);
  }}

  ctx.restore();
}}

// -------------------------------------------------------------
// Template: Comparison / Benchmark
// -------------------------------------------------------------
function drawComparison(t) {{
  const enter = easeOutExpo(Math.min(t / 0.7, 1.0));

  if (IS_VERTICAL) {{
    const blockW = W - 100;
    const blockH = (H - 420) / 2;
    const startX = 50;

    ctx.save();
    ctx.translate(startX, 240);
    ctx.scale(enter, enter);
    ctx.fillStyle = 'rgba(239, 68, 68, 0.08)';
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.4)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect(0, 0, blockW, blockH, 20);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#ef4444';
    ctx.font = 'bold 24px sans-serif';
    ctx.fillText('CÁCH TRUYỀN THỐNG (STALL)', 25, 45);

    const leftItems = ['Độ phức tạp O(N²)', 'Tắc nghẽn CPU Pipeline', 'Tốn 12 chu kỳ xử lý'];
    ctx.font = '19px sans-serif';
    ctx.fillStyle = '#fca5a5';
    leftItems.forEach((item, idx) => {{
      ctx.fillText('✗  ' + item, 25, 95 + idx * 45);
    }});
    ctx.restore();

    ctx.save();
    ctx.translate(startX, 240 + blockH + 30);
    ctx.scale(enter, enter);
    ctx.fillStyle = 'rgba(16, 185, 129, 0.12)';
    ctx.strokeStyle = GR;
    ctx.lineWidth = 2.5;
    ctx.shadowColor = GR;
    ctx.shadowBlur = 18;
    ctx.beginPath();
    ctx.roundRect(0, 0, blockW, blockH, 20);
    ctx.fill();
    ctx.stroke();
    ctx.shadowBlur = 0;

    ctx.fillStyle = GR;
    ctx.font = 'bold 24px sans-serif';
    ctx.fillText('KIẾN TRÚC TỐI ƯU (BYPASS)', 25, 45);

    const rightItems = ['Tối ưu O(N log N)', 'Truyền tắt Forwarding Path', 'Chỉ còn 3 chu kỳ thực thi'];
    ctx.font = 'bold 19px sans-serif';
    ctx.fillStyle = '#6ee7b7';
    rightItems.forEach((item, idx) => {{
      ctx.fillText('✓  ' + item, 25, 95 + idx * 45);
    }});

    const badgeEnter = easeOutExpo(Math.min(Math.max((t - 1.0) / 0.5, 0.0), 1.0));
    if (badgeEnter > 0.01) {{
      ctx.save();
      ctx.translate(blockW / 2, blockH - 45);
      ctx.scale(badgeEnter, badgeEnter);
      ctx.fillStyle = YE;
      ctx.shadowColor = YE;
      ctx.shadowBlur = 20;
      ctx.beginPath();
      ctx.roundRect(-90, -22, 180, 44, 22);
      ctx.fill();
      ctx.fillStyle = '#0f172a';
      ctx.font = 'bold 20px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('TỐC ĐỘ GẤP 4X', 0, 7);
      ctx.restore();
    }}
    ctx.restore();
    return;
  }}

  const colW = (W - 320) / 2;
  const colH = H - 280;
  const colY = 220;

  ctx.save();
  ctx.translate(120, colY);
  ctx.scale(enter, enter);
  ctx.fillStyle = 'rgba(239, 68, 68, 0.08)';
  ctx.strokeStyle = 'rgba(239, 68, 68, 0.4)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.roundRect(0, 0, colW, colH, 20);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = '#ef4444';
  ctx.font = 'bold 26px sans-serif';
  ctx.fillText('CÁCH TRUYỀN THỐNG (STALL)', 30, 50);

  const leftItems = ['Độ phức tạp O(N²)', 'Tắc nghẽn CPU Pipeline', 'Tốn 12 chu kỳ xử lý'];
  ctx.font = '20px sans-serif';
  ctx.fillStyle = '#fca5a5';
  leftItems.forEach((item, idx) => {{
    ctx.fillText('✗  ' + item, 30, 110 + idx * 50);
  }});
  ctx.restore();

  ctx.save();
  ctx.translate(120 + colW + 80, colY);
  ctx.scale(enter, enter);
  ctx.fillStyle = 'rgba(16, 185, 129, 0.12)';
  ctx.strokeStyle = GR;
  ctx.lineWidth = 2.5;
  ctx.shadowColor = GR;
  ctx.shadowBlur = 18;
  ctx.beginPath();
  ctx.roundRect(0, 0, colW, colH, 20);
  ctx.fill();
  ctx.shadowBlur = 0;

  ctx.fillStyle = GR;
  ctx.font = 'bold 26px sans-serif';
  ctx.fillText('KIẾN TRÚC TỐI ƯU (BYPASS)', 30, 50);

  const rightItems = ['Tối ưu O(N log N)', 'Truyền tắt Forwarding Path', 'Chỉ còn 3 chu kỳ thực thi'];
  ctx.font = 'bold 20px sans-serif';
  ctx.fillStyle = '#6ee7b7';
  rightItems.forEach((item, idx) => {{
    ctx.fillText('✓  ' + item, 30, 110 + idx * 50);
  }});

  const badgeEnter = easeOutExpo(Math.min(Math.max((t - 1.0) / 0.5, 0.0), 1.0));
  if (badgeEnter > 0.01) {{
    ctx.save();
    ctx.translate(colW / 2, colH - 60);
    ctx.scale(badgeEnter, badgeEnter);
    ctx.fillStyle = YE;
    ctx.shadowColor = YE;
    ctx.shadowBlur = 20;
    ctx.beginPath();
    ctx.roundRect(-100, -26, 200, 52, 26);
    ctx.fill();

    ctx.fillStyle = '#0f172a';
    ctx.font = 'bold 24px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('TỐC ĐỘ GẤP 4X', 0, 8);
    ctx.restore();
  }}

  ctx.restore();
}}

// -------------------------------------------------------------
// Template: Tech HUD / Metric Pulse
// -------------------------------------------------------------
function drawHUD(t) {{
  const cx = W / 2;
  const cy = IS_VERTICAL ? H / 2 : H / 2 + 60;
  const r = IS_VERTICAL ? 140 : 180;

  ctx.save();
  ctx.translate(cx, cy);

  ctx.strokeStyle = 'rgba(0, 240, 255, 0.2)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(0, 0, r, 0, Math.PI * 2);
  ctx.stroke();

  ctx.rotate(t * 1.2);
  ctx.strokeStyle = CY;
  ctx.lineWidth = 4;
  ctx.shadowColor = CY;
  ctx.shadowBlur = 15;
  ctx.beginPath();
  ctx.arc(0, 0, r + 25, 0, Math.PI * 0.7);
  ctx.stroke();

  ctx.strokeStyle = PK;
  ctx.shadowColor = PK;
  ctx.beginPath();
  ctx.arc(0, 0, r + 25, Math.PI, Math.PI * 1.6);
  ctx.stroke();
  ctx.restore();

  const p = Math.min(t / (DURATION * 0.8), 1.0);
  const val = Math.round(p * 1000000);

  ctx.save();
  ctx.textAlign = 'center';
  ctx.fillStyle = '#f8fafc';
  ctx.font = IS_VERTICAL ? 'bold 50px monospace' : 'bold 64px monospace';
  ctx.shadowColor = CY;
  ctx.shadowBlur = 20;
  ctx.fillText(val.toLocaleString() + '+', cx, cy + 15);
  ctx.shadowBlur = 0;

  ctx.fillStyle = '#94a3b8';
  ctx.font = IS_VERTICAL ? 'bold 18px sans-serif' : 'bold 22px sans-serif';
  ctx.fillText('REQUESTS / GIÂY XỬ LÝ', cx, cy + (IS_VERTICAL ? 55 : 65));
  ctx.restore();
}}

// -------------------------------------------------------------
// Master Render Function
// -------------------------------------------------------------
function renderFrameAt(f) {{
  const t = f / FPS;
  drawBackground(t);
  drawHeader(t);

  if (TEMPLATE === 'terminal') {{
    drawTerminal(t);
  }} else if (TEMPLATE === 'comparison') {{
    drawComparison(t);
  }} else if (TEMPLATE === 'hud') {{
    drawHUD(t);
  }} else {{
    drawArchitecture(t);
  }}
}}

// -------------------------------------------------------------
// Frame Extraction Loop qua HTTP POST
// -------------------------------------------------------------
async function runCapture() {{
  for (let f = 0; f < TOTAL_FRAMES; f++) {{
    renderFrameAt(f);
    const blob = await new Promise(res => canvas.toBlob(res, 'image/jpeg', 0.90));
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
        "Motion Graphics: Đã thu được %d/%d frames trong %.2fs",
        len(captured_frames), total_expected, dt,
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
