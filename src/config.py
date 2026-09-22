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
