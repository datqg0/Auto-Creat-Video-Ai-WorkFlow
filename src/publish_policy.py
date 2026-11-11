"""Machine approval cho luồng private -> public, không cần người duyệt thường xuyên."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import get_runtime_policy


@dataclass(slots=True)
class PublishDecision:
    approved: bool
    action: str
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"approved": self.approved, "action": self.action, "reasons": self.reasons}


def evaluate_publish_policy(
    manifest: dict[str, Any],
    quality_report: dict[str, Any],
    content_report: dict[str, Any],
) -> PublishDecision:
    """Chỉ approve khi mọi cổng blocking đều đã pass."""
    policy = get_runtime_policy().publishing
    reasons: list[str] = []
    if policy.mode != "automatic" or not policy.auto_publish_on_pass:
        reasons.append("Automatic publishing đang tắt.")
        return PublishDecision(False, "keep_private", reasons)
    if not content_report.get("passed", False):
        reasons.append("Content quality gate chưa pass.")
    if not quality_report.get("passed", False):
        reasons.append("Final video quality gate chưa pass.")
    if not manifest.get("quality_passed", False):
        reasons.append("Manifest không xác nhận quality_passed.")
    if not manifest.get("title") or not manifest.get("topic"):
        reasons.append("Manifest thiếu title hoặc topic.")
    if quality_report.get("blocking_errors"):
        reasons.append("Quality report còn blocking errors.")
    if content_report.get("errors"):
        reasons.append("Content report còn blocking errors.")
    if reasons:
        return PublishDecision(False, "quarantine", reasons)
    return PublishDecision(True, "publish", ["Tất cả machine quality gates đã pass."])
