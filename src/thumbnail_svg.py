"""Thumbnail SVG phong cách Minimalist Tech Architecture (chuẩn theo mẫu thiết kế).

Bố cục:
  - Nền tối sâu (radial gradient tối + viền neon mờ).
  - Chữ nhỏ gọn, tinh tế trong Glassmorphism Card (tiêu đề keyword + subtitle).
  - Core Node công nghệ bên trái (bát giác 3D + aperture ring phát sáng neon).
  - Laser Beam phát sáng kéo dài sang bên phải.
  - Cột các service icons tinh xảo (database, calendar, chip, tools...).
  - Render ra PNG độ nét cao (1280x720) bằng Microsoft Edge headless.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path

from PIL import Image

from .config import CONFIG
from .models import Script

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
_EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"


def _extract_thumb_meta(script: Script) -> dict:
    """Nhờ LLM trích xuất keyword ngắn và phụ đề cho thumbnail công nghệ."""
    subject = script.topic or script.title
    title = script.title

    try:
        from .llm import generate_text

        prompt = (
            "Phân tích tiêu đề video sau để trích xuất nội dung cho thumbnail tối giản kiểu công nghệ cao:\n"
            f"Tiêu đề: {title}\n"
            f"Chủ đề: {subject}\n\n"
            "YÊU CẦU BẮT BUỘC:\n"
            "1. keyword: Từ khóa công nghệ CHÍNH, viết HOA, cực ngắn (1-3 từ, ví dụ: 'MCP', 'DOCKER', 'KAFKA', 'TRANSFORMER', 'GRAPHQL', 'REDIS', 'NEURAL NET').\n"
            "2. subtitle: Phụ đề ngắn gọn, sâu sắc, viết HOA (2-5 từ, ví dụ: 'MỘT GIAO THỨC CHUNG', 'BẢN CHẤT HOẠT ĐỘNG', 'BÍ MẬT KIẾN TRÚC', 'CƠ CHẾ LÕI').\n"
            "3. source_label: Nhãn nguồn (1-3 từ, viết HOA, ví dụ: 'ỨNG DỤNG AI', 'CLIENT', 'MODEL', 'NGƯỜI DÙNG').\n"
            "4. theme: Chọn 1 màu neon hợp chủ đề ('purple' | 'cyan' | 'emerald' | 'amber').\n\n"
            "Trả về DUY NHẤT một JSON hợp lệ (không kèm markdown/lời giải thích):\n"
            '{"keyword": "...", "subtitle": "...", "source_label": "...", "theme": "..."}'
        )
        raw = generate_text(prompt, max_tokens=256).strip()
        if "```" in raw:
            parts = raw.split("```")
            for i in range(1, len(parts), 2):
                blk = parts[i].strip()
                if blk.startswith("json"):
                    blk = blk[4:].strip()
                if blk.startswith("{"):
                    raw = blk
                    break
        data = json.loads(raw)
        kw_str = str(data.get("keyword", "TECH")).upper().rstrip(":").strip()[:15]
        return {
            "keyword": kw_str or "TECH",
            "subtitle": str(data.get("subtitle", "KIẾN TRÚC HOẠT ĐỘNG")).upper()[:30],
            "source_label": str(data.get("source_label", "ỨNG DỤNG AI")).upper()[:20],
            "theme": str(data.get("theme", "purple")).lower(),
        }
    except Exception as e:
        log.warning("LLM trích xuất thumb meta lỗi (%s), dùng fallback", e)
        words = title.split()
        kw = words[0].rstrip(":").strip() if words else "TECH"
        return {
            "keyword": kw.upper()[:12] or "TECH",
            "subtitle": "BẢN CHẤT HOẠT ĐỘNG",
            "source_label": "ỨNG DỤNG AI",
            "theme": "purple",
        }


def _get_theme_palette(theme: str) -> dict:
    palettes = {
        "purple": {
            "primary": "#a855f7",
            "glow": "rgba(168, 85, 247, 0.7)",
            "beam_start": "#c084fc",
            "beam_end": "#7c3aed",
            "accent_icon": "#fbbf24",
        },
        "cyan": {
            "primary": "#38bdf8",
            "glow": "rgba(56, 189, 248, 0.7)",
            "beam_start": "#7dd3fc",
            "beam_end": "#0284c7",
            "accent_icon": "#34d399",
        },
        "emerald": {
            "primary": "#34d399",
            "glow": "rgba(52, 211, 153, 0.7)",
            "beam_start": "#6ee7b7",
            "beam_end": "#059669",
            "accent_icon": "#38bdf8",
        },
        "amber": {
            "primary": "#fbbf24",
            "glow": "rgba(251, 191, 36, 0.7)",
            "beam_start": "#fde68a",
            "beam_end": "#d97706",
            "accent_icon": "#f43f5e",
        },
    }
    return palettes.get(theme, palettes["purple"])


def build_svg_html(meta: dict) -> str:
    """Tạo mã HTML/SVG giao diện thumbnail chuẩn xác theo ảnh mẫu."""
    kw = meta["keyword"]
    sub = meta["subtitle"]
    src_label = meta["source_label"]
    color = _get_theme_palette(meta.get("theme", "purple"))

    # SVG HTML code
    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');
  
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    width: 1280px;
    height: 720px;
    background: radial-gradient(circle at 35% 45%, #161c2d 0%, #0a0d15 60%, #040508 100%);
    overflow: hidden;
    position: relative;
    font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
  }}

  /* Background Ambient Glow */
  .ambient-glow {{
    position: absolute;
    width: 600px;
    height: 600px;
    left: 100px;
    top: 80px;
    background: radial-gradient(circle, {color['glow']} 0%, transparent 65%);
    opacity: 0.18;
    filter: blur(80px);
    pointer-events: none;
  }}

  /* Top Card Container */
  .title-card {{
    position: absolute;
    top: 100px;
    left: 360px;
    padding: 20px 48px;
    background: rgba(18, 24, 38, 0.75);
    border: 1.5px solid rgba(255, 255, 255, 0.12);
    border-radius: 20px;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(16px);
    text-align: center;
    min-width: 320px;
  }}

  .title-keyword {{
    font-size: 54px;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: 2px;
    line-height: 1.1;
    font-family: 'Space Grotesk', sans-serif;
    text-shadow: 0 2px 10px rgba(0,0,0,0.5);
  }}

  .title-sub {{
    font-size: 20px;
    font-weight: 600;
    color: #94a3b8;
    letter-spacing: 2px;
    margin-top: 8px;
  }}

  /* Main Diagram Layer */
  svg.diagram {{
    position: absolute;
    top: 0;
    left: 0;
    width: 1280px;
    height: 720px;
  }}

  /* Source Label Pill */
  .source-pill {{
    position: absolute;
    top: 485px;
    left: 235px;
    padding: 10px 24px;
    background: rgba(15, 20, 32, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.14);
    border-radius: 12px;
    color: #e2e8f0;
    font-size: 20px;
    font-weight: 700;
    letter-spacing: 1.5px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.5);
    font-family: 'Space Grotesk', sans-serif;
  }}
</style>
</head>
<body>
  <div class="ambient-glow"></div>

  <!-- Header Box -->
  <div class="title-card">
    <div class="title-keyword">{kw}</div>
    <div class="title-sub">{sub}</div>
  </div>

  <!-- Source Label Pill -->
  <div class="source-pill">{src_label}</div>

  <!-- Main SVG Graphic Elements -->
  <svg class="diagram" viewBox="0 0 1280 720" fill="none" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <!-- Laser Beam Glow Filter -->
      <filter id="laser-glow" x="-20%" y="-150%" width="140%" height="400%">
        <feGaussianBlur in="SourceGraphic" stdDeviation="6" result="blur1" />
        <feGaussianBlur in="SourceGraphic" stdDeviation="18" result="blur2" />
        <feGaussianBlur in="SourceGraphic" stdDeviation="35" result="blur3" />
        <feMerge>
          <feMergeNode in="blur3" />
          <feMergeNode in="blur2" />
          <feMergeNode in="blur1" />
          <feMergeNode in="SourceGraphic" />
        </feMerge>
      </filter>

      <!-- Radial Gradients -->
      <linearGradient id="beam-grad" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%" stop-color="{color['beam_start']}" />
        <stop offset="60%" stop-color="{color['beam_end']}" />
        <stop offset="100%" stop-color="{color['beam_start']}" />
      </linearGradient>

      <!-- Octagon Dark Metal Gradient -->
      <linearGradient id="oct-grad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#2a3346" />
        <stop offset="50%" stop-color="#141824" />
        <stop offset="100%" stop-color="#0b0e17" />
      </linearGradient>
    </defs>

    <!-- LASER BEAM (From Core Node to Target Nodes) -->
    <!-- Soft Outer Halo -->
    <path d="M 410 375 L 1050 410" stroke="{color['primary']}" stroke-width="26" stroke-linecap="round" opacity="0.4" filter="url(#laser-glow)" />
    <!-- Bright Solid Core Beam -->
    <path d="M 410 375 L 1050 410" stroke="url(#beam-grad)" stroke-width="14" stroke-linecap="round" filter="url(#laser-glow)" />
    <!-- Internal White Intense Core -->
    <path d="M 410 375 L 1040 410" stroke="#ffffff" stroke-width="4" stroke-linecap="round" opacity="0.9" />

    <!-- CORE NODE (Center at X: 320, Y: 375) -->
    <g transform="translate(320, 375)">
      <!-- Outer Drop Shadow -->
      <polygon points="58,-140 140,-58 140,58 58,140 -58,140 -140,58 -140,-58 -58,-140" fill="#000000" opacity="0.6" filter="blur(20px)" />
      
      <!-- Outer Octagon Body (3D Metal Bevel) -->
      <polygon points="50,-120 120,-50 120,50 50,120 -50,120 -120,50 -120,-50 -50,-120" fill="url(#oct-grad)" stroke="rgba(255,255,255,0.18)" stroke-width="3" />
      <polygon points="45,-108 108,-45 108,45 45,108 -45,108 -108,45 -108,-45 -45,-108" fill="#10141f" stroke="rgba(0,0,0,0.8)" stroke-width="2" />

      <!-- Iris / Aperture Ring Outer -->
      <circle cx="0" cy="0" r="70" stroke="rgba(255,255,255,0.2)" stroke-width="3" fill="#0d111a" />
      <!-- Glowing Iris Inner -->
      <circle cx="0" cy="0" r="54" stroke="{color['primary']}" stroke-width="4" stroke-dasharray="14 8" opacity="0.9" />
      
      <!-- Iris Blades -->
      <line x1="-38" y1="-38" x2="38" y2="38" stroke="rgba(255,255,255,0.3)" stroke-width="2" />
      <line x1="0" y1="-54" x2="0" y2="54" stroke="rgba(255,255,255,0.3)" stroke-width="2" />
      <line x1="-54" y1="0" x2="54" y2="0" stroke="rgba(255,255,255,0.3)" stroke-width="2" />

      <!-- Center Intense Bright Pearl -->
      <circle cx="0" cy="0" r="28" fill="{color['primary']}" opacity="0.6" filter="url(#laser-glow)" />
      <circle cx="0" cy="0" r="18" fill="#ffffff" />
    </g>

    <!-- RIGHT TARGET ICONS (Service Ecosystem) -->
    <!-- 1. Top Icon: Tools / Document Container (X: 750, Y: 180) -->
    <g transform="translate(750, 180)" stroke="{color['accent_icon']}" stroke-width="3.5" fill="none" stroke-linecap="round" stroke-linejoin="round">
      <rect x="0" y="20" width="60" height="40" rx="8" />
      <path d="M12 20 L24 6 L48 6 L56 20" />
      <line x1="20" y1="36" x2="40" y2="36" />
    </g>

    <!-- 2. Mid-Right Icon: Database Cylinder (X: 970, Y: 220) -->
    <g transform="translate(970, 220)" stroke="#38bdf8" stroke-width="3.5" fill="none" stroke-linecap="round" stroke-linejoin="round">
      <ellipse cx="30" cy="14" rx="30" ry="11" />
      <path d="M0 14 v20 c0 6 13 11 30 11 s30 -5 30 -11 v-20" />
      <path d="M0 34 v20 c0 6 13 11 30 11 s30 -5 30 -11 v-20" />
    </g>

    <!-- 3. Bottom-Left Target Icon: Chip / Microcontroller (X: 750, Y: 460) -->
    <g transform="translate(750, 460)" stroke="#94a3b8" stroke-width="3.5" fill="none" stroke-linecap="round" stroke-linejoin="round">
      <rect x="10" y="10" width="48" height="48" rx="10" />
      <circle cx="26" cy="26" r="4" fill="#94a3b8" />
      <!-- Pins -->
      <line x1="58" y1="24" x2="68" y2="24" />
      <line x1="58" y1="44" x2="68" y2="44" />
      <line x1="0" y1="34" x2="10" y2="34" />
    </g>

    <!-- 4. Bottom-Right Target Icon: Calendar / Scheduler (X: 980, Y: 460) -->
    <g transform="translate(980, 460)" stroke="#f43f5e" stroke-width="3.5" fill="none" stroke-linecap="round" stroke-linejoin="round">
      <rect x="0" y="12" width="56" height="52" rx="10" />
      <line x1="0" y1="28" x2="56" y2="28" />
      <!-- Ring binders -->
      <line x1="14" y1="4" x2="14" y2="14" stroke-width="4" />
      <line x1="42" y1="4" x2="42" y2="14" stroke-width="4" />
      <!-- Checkmark inside -->
      <path d="M18 44 l7 7 l16 -16" stroke-width="3" />
    </g>
  </svg>
</body>
</html>
"""


def render_svg_thumbnail(script: Script, out_path: Path) -> Path | None:
    """Tạo thumbnail SVG chuẩn phong cách kiến trúc công nghệ và xuất PNG qua msedge."""
    meta = _extract_thumb_meta(script)
    html_content = build_svg_html(meta)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_html = out_path.parent / "_temp_thumb.html"
    temp_html.write_text(html_content, encoding="utf-8")

    # Render bằng Microsoft Edge headless
    if os.path.exists(_EDGE_PATH):
        try:
            cmd = [
                _EDGE_PATH,
                "--headless",
                "--disable-gpu",
                f"--screenshot={out_path.resolve()}",
                "--window-size=1280,720",
                "--hide-scrollbars",
                str(temp_html.resolve()),
            ]
            res = subprocess.run(cmd, timeout=30, capture_output=True, text=True)
            if res.returncode == 0 and out_path.exists() and out_path.stat().st_size > 1024:
                log.info("Đã tạo thumbnail SVG công nghệ sắc nét: %s", out_path)
                try:
                    temp_html.unlink(missing_ok=True)
                except Exception:
                    pass
                return out_path
        except Exception as e:
            log.warning("Render Edge headless lỗi: %s", e)

    # Dọn dẹp
    try:
        temp_html.unlink(missing_ok=True)
    except Exception:
        pass
    return None
