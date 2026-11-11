"""Đọc config.yaml và biến môi trường."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

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


class ManimSandboxPolicy(BaseModel):
    backend: Literal["local", "container"] = "local"
    image: str = "tech-viz-manim:latest"
    network: bool = False
    memory_mb: int = Field(default=4096, ge=512, le=16384)
    cpus: float = Field(default=2.0, gt=0, le=16)
    pids_limit: int = Field(default=128, ge=16, le=1024)
    timeout_seconds: int = Field(default=480, ge=30, le=3600)


class ContentQualityPolicy(BaseModel):
    enabled: bool = True
    minimum_score: float = Field(default=82.0, ge=0.0, le=100.0)
    minimum_accuracy_score: float = Field(default=27.0, ge=0.0, le=30.0)
    require_references: bool = True
    llm_critic_enabled: bool = True
    block_on_llm_errors: bool = False
    auto_revise: bool = True
    max_revision_attempts: int = Field(default=1, ge=0, le=2)


class PublishingPolicy(BaseModel):
    mode: Literal["manual", "automatic"] = "automatic"
    staging_privacy: Literal["private", "unlisted"] = "private"
    final_privacy: Literal["public", "unlisted"] = "public"
    auto_publish_on_pass: bool = True
    processing_timeout_seconds: int = Field(default=1800, ge=60, le=21600)
    poll_interval_seconds: int = Field(default=30, ge=5, le=300)
    max_retries: int = Field(default=2, ge=0, le=10)


class RuntimePolicy(BaseModel):
    environment: Literal["local", "production"] = "local"
    manim: ManimSandboxPolicy = Field(default_factory=ManimSandboxPolicy)
    content: ContentQualityPolicy = Field(default_factory=ContentQualityPolicy)
    publishing: PublishingPolicy = Field(default_factory=PublishingPolicy)


def get_runtime_policy() -> RuntimePolicy:
    """Đọc phần policy có kiểu mà không thay đổi CONFIG dict tương thích cũ."""
    animation = CONFIG.get("animation", {}) or {}
    return RuntimePolicy.model_validate({
        "environment": (env("APP_ENV") or (CONFIG.get("runtime", {}) or {}).get("environment") or "local").lower(),
        "manim": animation.get("sandbox", {}) or {},
        "content": CONFIG.get("content_quality", {}) or {},
        "publishing": CONFIG.get("publishing", {}) or {},
    })


def validate_runtime_config() -> RuntimePolicy:
    """Fail closed với các bất biến bắt buộc khi chạy production."""
    policy = get_runtime_policy()
    if policy.environment == "production":
        database_url = env("DATABASE_URL") or ""
        if not database_url.startswith(("postgresql://", "postgres://")):
            raise RuntimeError("Production bắt buộc DATABASE_URL PostgreSQL; không dùng SQLite fallback.")
        if policy.manim.backend != "container":
            raise RuntimeError("Production bắt buộc animation.sandbox.backend=container.")
        if policy.publishing.staging_privacy != "private":
            raise RuntimeError("Production tự động phải upload staging ở trạng thái private.")
        animation = CONFIG.get("animation", {}) or {}
        if animation.get("matplotlib_fallback_enabled", False):
            raise RuntimeError("Production không cho phép thực thi Matplotlib code do LLM sinh.")
    return policy


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
