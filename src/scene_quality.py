"""Kiểm tra asset từng scene trước khi composite toàn video."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat

from .config import CONFIG
from .models import Scene


def audit_scene_assets(
    scenes: list[Scene],
    images: list[Path],
    code_videos: list[Path | None],
    broll_videos: list[Path | None],
    renderers: list[str],
    output_path: Path,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    expected_size = (
        int((CONFIG.get("visual", {}) or {}).get("width", 1920)),
        int((CONFIG.get("visual", {}) or {}).get("height", 1080)),
    )
    min_video_bytes = int((CONFIG.get("scene_quality", {}) or {}).get("min_animation_bytes", 4096))

    for index, scene in enumerate(scenes):
        image = images[index] if index < len(images) else None
        if image is None or not image.exists():
            checks.append({"scene": index, "name": "fallback_image", "ok": False, "severity": "error", "message": "Thiếu ảnh fallback."})
        else:
            try:
                with Image.open(image) as frame:
                    size_ok = frame.size == expected_size
                    checks.append({
                        "scene": index, "name": "fallback_resolution", "ok": size_ok,
                        "severity": "error", "message": f"{frame.size}; yêu cầu {expected_size}.",
                    })
                    stat = ImageStat.Stat(frame.convert("L"))
                    variance = float(stat.var[0]) if stat.var else 0.0
                    checks.append({
                        "scene": index, "name": "visual_variance", "ok": variance >= 8.0,
                        "severity": "warning", "message": f"Variance={variance:.2f}.",
                    })
            except Exception as exc:  # noqa: BLE001
                checks.append({"scene": index, "name": "fallback_read", "ok": False, "severity": "error", "message": str(exc)})

        animation = code_videos[index] if index < len(code_videos) else None
        renderer = renderers[index] if index < len(renderers) else "unknown"
        if renderer in {"manim", "motion_graphics", "matplotlib_ai", "generated_animation"}:
            ok = bool(animation and animation.exists() and animation.stat().st_size >= min_video_bytes)
            checks.append({
                "scene": index, "name": "animation_file", "ok": ok, "severity": "error",
                "message": f"renderer={renderer}; path={animation}",
            })
        if renderer == "broll":
            broll = broll_videos[index] if index < len(broll_videos) else None
            ok = bool(broll and broll.exists() and broll.stat().st_size >= min_video_bytes)
            checks.append({
                "scene": index, "name": "broll_file", "ok": ok, "severity": "error",
                "message": f"path={broll}",
            })

        contract = scene.contract
        if contract:
            visible_chars = len(scene.heading) + sum(len(str(item)) for item in scene.bullets)
            checks.append({
                "scene": index, "name": "text_budget", "ok": visible_chars <= contract.max_on_screen_chars,
                "severity": "warning",
                "message": f"{visible_chars}/{contract.max_on_screen_chars} ký tự dự kiến trên màn hình.",
            })

    errors = [item for item in checks if not item["ok"] and item["severity"] == "error"]
    warnings = [item for item in checks if not item["ok"] and item["severity"] == "warning"]
    fail_on_warnings = bool((CONFIG.get("scene_quality", {}) or {}).get("fail_on_warnings", False))
    report = {
        "passed": not errors and (not warnings or not fail_on_warnings),
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def require_scene_quality_pass(report: dict[str, Any], report_path: Path) -> None:
    if report.get("passed"):
        return
    messages = [str(item.get("message")) for item in report.get("errors", [])]
    raise RuntimeError(f"Scene QA thất bại; xem {report_path}: " + "; ".join(messages))
