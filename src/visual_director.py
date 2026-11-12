"""Lập kế hoạch hình ảnh có cấu trúc trước khi gọi renderer."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .config import CONFIG
from .models import Scene, SceneContract, Script, TimelineCue


_MANIM_TERMS = (
    "algorithm", "thuật toán", "matrix", "ma trận", "vector", "graph", "đồ thị",
    "tree", "cây", "binary", "sorting", "sắp xếp", "pointer", "dp", "stack",
    "queue", "hàm số", "đạo hàm", "tích phân", "neural", "attention", "gradient",
)
_MOTION_TERMS = (
    "kiến trúc", "architecture", "pipeline", "workflow", "timeline", "so sánh",
    "database", "server", "client", "network", "mạng", "flow", "quy trình",
)


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?…])\s+", text or "") if part.strip()]


def _renderer_for(scene: Scene) -> str:
    corpus = " ".join((scene.heading, scene.visual_prompt, scene.algorithm, scene.narration)).lower()
    preset = str((scene.animation or {}).get("preset", "")).lower()
    if scene.visual_type in {"math", "math_equation", "algorithm"}:
        return "manim"
    if any(term in corpus for term in _MOTION_TERMS):
        return "motion_graphics"
    if scene.visual_type == "animation":
        return "manim"
    if any(term in corpus for term in _MANIM_TERMS):
        return "manim"
    if scene.visual_type in {"diagram", "chart", "code"}:
        return "motion_graphics"
    if scene.video_query or scene.image_query:
        return "broll"
    return "static"


def _contract_for(scene: Scene) -> SceneContract:
    renderer = _renderer_for(scene)
    fallbacks = {
        "manim": ["motion_graphics", "static"],
        "motion_graphics": ["static"],
        "broll": ["motion_graphics", "static"],
        "static": ["static"],
    }[renderer]
    sentences = _sentences(scene.narration)
    timeline = [
        TimelineCue(
            at=min(0.95, index / max(1, len(sentences))),
            action="reveal" if index == 0 else "advance",
            target=(scene.bullets[index] if index < len(scene.bullets) else f"beat_{index + 1}")[:80],
        )
        for index, _ in enumerate(sentences[:6])
    ]
    objects = [str(item)[:80] for item in (scene.bullets or [])[:6]]
    if not objects and scene.heading:
        objects = [scene.heading[:80]]
    teaching_goal = scene.heading or (sentences[0] if sentences else "Minh họa luận điểm của cảnh")
    return SceneContract(
        teaching_goal=teaching_goal[:180],
        preferred_renderer=renderer,
        fallback_renderers=fallbacks,
        objects=objects,
        focus=(objects[0] if objects else scene.heading)[:80],
        timeline=timeline,
        claim_ids=list(scene.claim_ids),
        max_on_screen_chars=int((CONFIG.get("visual_style", {}) or {}).get("max_text_per_frame", 70)),
        safe_margin_percent=float((CONFIG.get("visual_style", {}) or {}).get("safe_margin_percent", 8)),
        subtitle_safe_area_percent=float((CONFIG.get("visual_style", {}) or {}).get("subtitle_safe_area_percent", 18)),
    )


def plan_visuals(script: Script, output_path: Path | None = None) -> Script:
    """Gắn SceneContract cho từng cảnh và tùy chọn ghi storyboard JSON."""
    planned = script.model_copy(deep=True)
    animation_cfg = CONFIG.get("animation", {}) or {}
    active_mode = str(CONFIG.get("active_mode") or "long")
    for index, scene in enumerate(planned.scenes):
        scene.contract = _contract_for(scene)
        if index == 0 and active_mode != "documentary" and animation_cfg.get("animate_from_start", True):
            scene.contract.preferred_renderer = "manim"
            scene.contract.fallback_renderers = ["motion_graphics", "static"]
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps([scene.contract.model_dump() for scene in planned.scenes], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return planned
