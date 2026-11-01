"""Lightweight, deterministic checks for generated scripts."""
from __future__ import annotations

import json
import re
from collections import Counter

from .models import Script


_MODULAR_ARITHMETIC = re.compile(
    r"(?P<base>\d+)\s*(?:\^|\*\*)\s*(?P<exp>\d+)\s*"
    r"(?:mod(?:ulo)?)\s*(?P<modulus>\d+)\s*=\s*(?P<result>\d+)",
    re.IGNORECASE,
)


def inspect_script(script: Script) -> dict:
    """Return blocking errors and review warnings without calling an external model."""
    errors: list[str] = []
    warnings: list[str] = []
    if not script.title.strip():
        errors.append("Kịch bản thiếu tiêu đề.")
    if len(script.scenes) < 3:
        errors.append(f"Chỉ có {len(script.scenes)} cảnh; cần ít nhất 3 cảnh.")

    seen: Counter[str] = Counter()
    for index, scene in enumerate(script.scenes, start=1):
        narration = " ".join((scene.narration or "").lower().split())
        if not narration:
            errors.append(f"Cảnh {index} không có lời đọc.")
            continue
        seen[narration] += 1
        visual_text = " ".join([
            str(scene.heading or ""),
            " ".join(str(item) for item in (scene.bullets or [])),
            str(scene.visual_prompt or ""),
            json.dumps(scene.animation or {}, ensure_ascii=False),
        ])
        expressions = {
            match.group(0): match
            for match in _MODULAR_ARITHMETIC.finditer(f"{scene.narration or ''} {visual_text}")
        }
        for match in expressions.values():
            base = int(match.group("base"))
            exponent = int(match.group("exp"))
            modulus = int(match.group("modulus"))
            claimed = int(match.group("result"))
            if modulus <= 0 or pow(base, exponent, modulus) != claimed:
                actual = "không xác định" if modulus <= 0 else str(pow(base, exponent, modulus))
                errors.append(
                    f"Cảnh {index}: phép tính {match.group(0)!r} sai; kết quả đúng là {actual}."
                )

    for narration, count in seen.items():
        if count > 1 and len(narration) >= 50:
            warnings.append(f"Có lời đọc lặp nguyên văn {count} lần: {narration[:90]}...")
    if not script.references:
        warnings.append("Chưa có nguồn tham khảo công khai; cần nghe/đọc soát các khẳng định kỹ thuật.")

    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "scene_count": len(script.scenes),
        "reference_count": len(script.references),
    }
