"""Theme / bảng màu dùng chung cho mathviz.

Đồng bộ màu với config.yaml để animation khớp phong cách các scene tĩnh.
"""
from __future__ import annotations

from pathlib import Path

try:
    from ..config import CONFIG  # type: ignore
except Exception:  # noqa: BLE001 - chạy standalone (test) không có config
    CONFIG = None


def _cfg(path: list[str], default):
    node = CONFIG
    if node is None:
        return default
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


_ROOT = Path(__file__).resolve().parent.parent.parent


def _abs_font(rel: str) -> str:
    p = _ROOT / rel
    return str(p) if p.exists() else rel


class Theme:
    """Gom toàn bộ hằng số style vào 1 chỗ để chỉnh nhanh."""

    def __init__(self) -> None:
        self.width: int = _cfg(["visual", "width"], 1920)
        self.height: int = _cfg(["visual", "height"], 1080)
        self.fps: int = _cfg(["visual", "fps"], 30)

        self.bg: str = _cfg(["visual", "background_color"], "#0d1117")
        self.accent: str = _cfg(["visual", "accent_color"], "#58a6ff")

        self.text: str = "#e6edf3"
        self.muted: str = "#8b949e"
        self.panel: str = "#161b22"
        self.grid: str = "#21262d"

        # Bảng màu phụ, xoay vòng cho các phần tử.
        self.palette: list[str] = [
            "#58a6ff",  # xanh dương
            "#3fb950",  # xanh lá
            "#d29922",  # vàng
            "#f778ba",  # hồng
            "#a371f7",  # tím
            "#39c5cf",  # cyan
            "#ff7b72",  # đỏ cam
        ]

        self.font_bold: str = _abs_font(
            _cfg(["visual", "font"], "assets/fonts/BeVietnamPro-Bold.ttf")
        )
        self.font_regular: str = _abs_font(
            _cfg(["visual", "font_regular"], "assets/fonts/BeVietnamPro-Regular.ttf")
        )

    def color(self, index: int) -> str:
        """Lấy màu trong palette theo index (tự xoay vòng)."""
        return self.palette[index % len(self.palette)]


THEME = Theme()


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    c = color.lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16))


def hex_to_rgba(color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    r, g, b = hex_to_rgb(color)
    return (r, g, b, alpha)


def mix(c1: str, c2: str, t: float) -> tuple[int, int, int]:
    a, b = hex_to_rgb(c1), hex_to_rgb(c2)
    t = max(0.0, min(1.0, t))
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))  # type: ignore[return-value]
