"""AI tự chọn chủ đề tech, tránh trùng với các chủ đề đã dùng gần đây."""
from __future__ import annotations

import logging
import random

from . import db
from .config import CONFIG
from .llm import generate

log = logging.getLogger(__name__)


def pick_topic() -> str:
    topics_cfg = CONFIG["topics"]
    domains: list[str] = topics_cfg["domains"]
    dedup_days = int(topics_cfg.get("dedup_days", 30))

    used = db.recent_topics(dedup_days)
    domain = random.choice(domains)

    used_block = "\n".join(f"- {t}" for t in used) if used else "(chưa có)"
    lang = CONFIG.get("language", "vi")
    lang_name = "tiếng Việt" if lang == "vi" else "English"

    prompt = f"""Bạn là biên tập viên kênh YouTube về công nghệ theo phong cách giải thích trực quan (visualization).
Hãy đề xuất MỘT chủ đề video cụ thể, hấp dẫn trong lĩnh vực: {domain}.

Yêu cầu:
- Chủ đề đủ hẹp để giải thích trong video 5 phút bằng hình ảnh động/biểu đồ.
- Phù hợp để minh họa bằng animation, biểu đồ dữ liệu, hoặc mô phỏng thuật toán.
- Viết bằng {lang_name}.
- KHÔNG trùng hoặc quá giống các chủ đề đã dùng dưới đây:
{used_block}

Chỉ trả về DUY NHẤT tên chủ đề trên một dòng, không giải thích, không đánh số, không dấu ngoặc kép."""

    topic = generate(prompt).strip().splitlines()[0].strip().strip('"').strip("-").strip()
    log.info("Chủ đề đã chọn: %s", topic)
    return topic
