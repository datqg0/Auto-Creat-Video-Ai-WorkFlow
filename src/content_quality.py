"""Cổng chất lượng nội dung kết hợp kiểm tra tất định và critic LLM tùy chọn."""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from .config import get_runtime_policy
from .models import Script
from .script_quality import inspect_script

log = logging.getLogger(__name__)


def _sentence_lengths(text: str) -> list[int]:
    return [len(part.strip()) for part in re.split(r"(?<=[.!?…])\s+", text or "") if part.strip()]


def _deterministic_scores(script: Script, base_report: dict[str, Any]) -> tuple[dict[str, float], list[str]]:
    warnings: list[str] = []
    narrations = [scene.narration or "" for scene in script.scenes]
    corpus = " ".join(narrations).lower()

    accuracy = 30.0
    accuracy -= min(30.0, 10.0 * len(base_report.get("errors", [])))
    if not script.references:
        accuracy -= 5.0
    trusted_references = {url.rstrip("/") for url in script.references}
    unbacked_claims = [
        claim for claim in script.evidence
        if not claim.source_url
        or claim.source_url.rstrip("/") not in trusted_references
        or claim.confidence < 0.65
    ]
    accuracy -= min(8.0, len(unbacked_claims) * 2.0)
    if unbacked_claims:
        warnings.append(f"Có {len(unbacked_claims)} claim chưa khớp nguồn research hoặc confidence thấp.")
    known_claim_ids = {claim.claim_id for claim in script.evidence}
    unknown_claim_ids = {
        claim_id for scene in script.scenes for claim_id in scene.claim_ids
        if claim_id not in known_claim_ids
    }
    if unknown_claim_ids:
        accuracy -= min(5.0, float(len(unknown_claim_ids)))
        warnings.append(f"Scene tham chiếu claim_id chưa tồn tại: {sorted(unknown_claim_ids)[:5]}.")

    clarity = 20.0
    long_sentences = sum(length > 220 for text in narrations for length in _sentence_lengths(text))
    clarity -= min(10.0, long_sentences * 1.5)
    dense_scenes = sum(len(text) > 1300 for text in narrations)
    clarity -= min(6.0, dense_scenes * 2.0)
    if long_sentences:
        warnings.append(f"Có {long_sentences} câu quá dài, khó nghe bằng TTS.")

    logic = 15.0
    if len(script.scenes) < 5:
        logic -= 3.0
    if script.scenes and not any(mark in script.scenes[0].narration for mark in ("?", "vì sao", "điều gì", "hãy")):
        logic -= 2.0
    conclusion_words = ("tóm lại", "kết luận", "như vậy", "điểm mấu chốt", "cuối cùng")
    if script.scenes and not any(word in script.scenes[-1].narration.lower() for word in conclusion_words):
        logic -= 3.0

    practical = 15.0
    example_terms = ("ví dụ", "giả sử", "thực tế", "trường hợp", "hãy thử", "bài toán")
    if not any(term in corpus for term in example_terms):
        practical -= 6.0
        warnings.append("Kịch bản chưa có ví dụ hoặc tình huống thực tế rõ ràng.")
    if not script.exercises and str(os.environ.get("VIDEO_MODE", "long")) != "documentary":
        practical -= 2.0

    vietnamese = 10.0
    markdown_artifacts = sum(text.count("```") + text.count("**") for text in narrations)
    vietnamese -= min(5.0, markdown_artifacts)
    if markdown_artifacts:
        warnings.append("Lời đọc còn ký hiệu Markdown có thể làm TTS đọc không tự nhiên.")

    retention = 10.0
    question_scenes = sum("?" in text for text in narrations)
    if question_scenes == 0:
        retention -= 3.0
    if len(script.scenes) >= 6 and question_scenes < 2:
        retention -= 2.0
    visual_changes = len({scene.visual_type for scene in script.scenes})
    if len(script.scenes) >= 5 and visual_changes < 3:
        retention -= 2.0

    scores = {
        "accuracy": max(0.0, round(accuracy, 1)),
        "clarity": max(0.0, round(clarity, 1)),
        "logic": max(0.0, round(logic, 1)),
        "practical_value": max(0.0, round(practical, 1)),
        "vietnamese": max(0.0, round(vietnamese, 1)),
        "retention": max(0.0, round(retention, 1)),
    }
    return scores, warnings


def _extract_json_object(text: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Critic không trả JSON object.")
    raw = text[start:end + 1]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        from json_repair import loads

        value = loads(raw)
        if not isinstance(value, dict):
            raise ValueError("Critic JSON không phải object.")
        return value


def _llm_critic(script: Script) -> dict[str, Any] | None:
    policy = get_runtime_policy().content
    if not policy.llm_critic_enabled:
        return None
    try:
        from .llm import generate

        compact = script.model_dump(mode="json", exclude={"content_score"})
        prompt = (
            "Bạn là biên tập viên độc lập cho video giáo dục công nghệ tiếng Việt. "
            "Không viết lại kịch bản. Hãy tìm lỗi có thể hành động được, đặc biệt là fact, "
            "logic, cách giải thích cho người mới, tiếng Việt tự nhiên và retention. "
            "Chỉ trả JSON theo schema: "
            '{"blocking_errors":[],"warnings":[],"revision_instructions":[],"confidence":0.0}. '
            "Chỉ đưa blocking error khi có mâu thuẫn hoặc sai rõ ràng; không đoán nguồn.\n\n"
            + json.dumps(compact, ensure_ascii=False)
        )
        return _extract_json_object(generate(prompt, task="script"))
    except Exception as exc:  # noqa: BLE001 - critic lỗi không được làm mất deterministic QA
        log.warning("Content critic LLM không khả dụng: %s", exc)
        return {"blocking_errors": [], "warnings": [f"LLM critic không chạy được: {exc}"], "revision_instructions": []}


def evaluate_content(script: Script, output_path: Path | None = None) -> dict[str, Any]:
    """Chấm script trên thang 100 và trả policy decision có cấu trúc."""
    policy = get_runtime_policy().content
    base = inspect_script(script)
    scores, scoring_warnings = _deterministic_scores(script, base)
    total = round(sum(scores.values()), 1)
    critic = _llm_critic(script)

    errors = list(base.get("errors", []))
    warnings = list(base.get("warnings", [])) + scoring_warnings
    if critic:
        warnings.extend(str(item) for item in critic.get("warnings", []) if str(item).strip())
        if policy.block_on_llm_errors:
            errors.extend(str(item) for item in critic.get("blocking_errors", []) if str(item).strip())

    if policy.require_references and not script.references:
        errors.append("Content policy yêu cầu ít nhất một nguồn tham khảo công khai.")
    if scores["accuracy"] < policy.minimum_accuracy_score:
        errors.append(
            f"Điểm accuracy {scores['accuracy']:.1f}/30 thấp hơn ngưỡng {policy.minimum_accuracy_score:.1f}."
        )
    if total < policy.minimum_score:
        errors.append(f"Điểm nội dung {total:.1f}/100 thấp hơn ngưỡng {policy.minimum_score:.1f}.")

    report = {
        "passed": not errors,
        "score": total,
        "scores": scores,
        "errors": errors,
        "warnings": warnings,
        "critic": critic,
        "reference_count": len(script.references),
        "evidence_claim_count": len(script.evidence),
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def revise_content(script: Script, report: dict[str, Any]) -> Script | None:
    """Một lượt sửa có giới hạn; không được thay nguồn, topic hoặc metadata nhận diện."""
    policy = get_runtime_policy().content
    if not policy.auto_revise or policy.max_revision_attempts < 1 or not script.references:
        return None
    critic = report.get("critic") or {}
    instructions = [str(item) for item in critic.get("revision_instructions", []) if str(item).strip()]
    actionable = instructions or [str(item) for item in report.get("warnings", []) if str(item).strip()]
    if not actionable:
        return None
    # Không dùng revision để lách lỗi nguồn/fact tất định.
    hard_errors = " ".join(str(item).lower() for item in report.get("errors", []))
    if "nguồn tham khảo" in hard_errors or "phép tính" in hard_errors:
        return None
    try:
        from .llm import generate

        prompt = (
            "Bạn là script editor. Sửa đúng các vấn đề dưới đây nhưng giữ nguyên topic, title, references, "
            "số scene và mọi URL. Không thêm fact/số liệu mới. Giữ JSON schema tương thích Script. "
            "Chỉ trả một JSON object đầy đủ.\n\nVẤN ĐỀ:\n- "
            + "\n- ".join(actionable[:12])
            + "\n\nSCRIPT:\n"
            + script.model_dump_json(
                exclude={"content_score": True, "scenes": {"__all__": {"contract"}}}
            )
        )
        raw = generate(prompt, task="script")
        data = _extract_json_object(raw)
        revised = Script.model_validate(data)
        if len(revised.scenes) != len(script.scenes):
            raise ValueError("Revision thay đổi số scene.")
        return revised.model_copy(update={
            "topic": script.topic,
            "title": script.title,
            "references": list(script.references),
            "tags": list(script.tags),
        })
    except Exception as exc:  # noqa: BLE001
        log.warning("Không tự sửa được content: %s", exc)
        return None


def require_content_pass(report: dict[str, Any], workdir: Path) -> None:
    """Dừng trước render nếu content policy không đạt."""
    if report.get("passed"):
        return
    raise RuntimeError(
        f"Kịch bản không qua content gate; xem {workdir / 'content_quality_report.json'}: "
        + "; ".join(str(item) for item in report.get("errors", []))
    )
