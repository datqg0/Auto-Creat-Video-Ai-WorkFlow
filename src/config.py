"""Đọc config.yaml và biến môi trường."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"
OUTPUT_DIR = ROOT / "output"
ASSETS_DIR = ROOT / "assets"
CREDENTIALS_DIR = ROOT / "credentials"


def load_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def env(key: str, default: str | None = None) -> str | None:
    return os.getenv(key, default)


CONFIG = load_config()


def apply_mode(mode: str | None = None) -> str:
    """Áp cấu hình theo buổi (long/short) vào CONFIG tại chỗ.

    Ưu tiên: tham số mode > env VIDEO_MODE > config default_mode > "long".
    Ghi đè target_duration_seconds và visual.width/height. Trả về tên mode đã áp.
    Phải gọi TRƯỚC khi import các module cache W/H (visual_engine, compositor, mathviz).
    """
    modes = CONFIG.get("modes") or {}
    mode = mode or os.getenv("VIDEO_MODE") or CONFIG.get("default_mode") or "long"
    if mode not in modes:
        return mode
    m = modes[mode]
    if "target_duration_seconds" in m:
        CONFIG["target_duration_seconds"] = m["target_duration_seconds"]
    CONFIG.setdefault("visual", {})
    if "width" in m:
        CONFIG["visual"]["width"] = m["width"]
    if "height" in m:
        CONFIG["visual"]["height"] = m["height"]
    CONFIG["active_mode"] = mode
    os.environ["VIDEO_MODE"] = mode
    return mode
