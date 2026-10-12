"""AI tu chon chu de tech, tranh trung voi cac chu de da dung gan day.
Tich hop du lieu trending (HackerNews, Reddit, Google Trends) lam ngu canh.
Dam bao >=1/3 so video thuoc nhom: GitHub repos, thuat toan AI/ML, open-source.
"""
from __future__ import annotations

import logging
import random

from . import db
from .config import CONFIG

log = logging.getLogger(__name__)


def _should_use_priority_domain() -> bool:
    """Kiểm tra xem lần chọn này có nên dùng priority domain không.

    Logic: Cứ N video thì 1 video bắt buộc từ priority domain (GitHub/AI/thuật toán).
    Đếm tổng số video đã tạo gần đây trong dedup_days để quyết định.
    """
    topics_cfg = CONFIG["topics"]
    every_n = int(topics_cfg.get("priority_domains_every_n", 3))
    dedup_days = int(topics_cfg.get("dedup_days", 30))

    try:
        # Đếm video đã tạo trong N ngày gần nhất (bao gồm cả ngày hôm nay)
        recent = db.recent_topics(dedup_days)
        total = len(recent)
        # Nếu tổng video % every_n == 0 (ví dụ lần 0, 3, 6...) -> dùng priority
        return (total % every_n) == 0
    except Exception:
        # Fallback: 1/3 xác suất ngẫu nhiên
        return random.randint(1, 3) == 1


def pick_topic() -> str:
    topics_cfg = CONFIG["topics"]
    domains: list[str] = topics_cfg["domains"]
    dedup_days = int(topics_cfg.get("dedup_days", 30))
    priority_indices: list[int] = topics_cfg.get("priority_domain_indices", [])

    used = db.recent_topics(dedup_days)

    # Quyết định dùng priority domain hay random domain
    use_priority = priority_indices and _should_use_priority_domain()

    if use_priority:
        valid_priority = [i for i in priority_indices if 0 <= i < len(domains)]
        if valid_priority:
            domain = domains[random.choice(valid_priority)]
            log.info("🎯 Chọn PRIORITY domain (GitHub/AI/thuật toán): %s", domain)
        else:
            domain = random.choice(domains)
    else:
        domain = random.choice(domains)

    used_block = "\n".join(f"- {t}" for t in used) if used else "(chua co)"
    lang = CONFIG.get("language", "vi")
    lang_name = "tieng Viet" if lang == "vi" else "English"

    # --- Lay tin hieu trending (khong bat buoc; loi mang thi bo qua) ---
    trend_block = ""
    try:
        from .trend_fetcher import fetch_trends, format_for_prompt
        trends = fetch_trends()
        trend_block = format_for_prompt(trends, max_items=18)
    except Exception as e:  # noqa: BLE001 - trend la uu tien, khong phai bat buoc
        log.debug("Khong lay duoc trend: %s", e)

    trend_section = (
        f"\n\nXU HUONG CONG NGHE HIEN TAI (su dung de chon chu de phu hop thoi su):\n{trend_block}\n"
        "Hay THAM KHAO cac xu huong nay de chon chu de gan voi dieu nguoi dung dang quan tam,"
        " nhung KHONG chep nguyen tieu de bai viet. Bien doi thanh goc nhin giai thich truc quan."
        if trend_block else ""
    )

    # Thêm hướng dẫn đặc biệt nếu là priority domain GitHub/AI
    github_hint = ""
    if "github" in domain.lower() or "open-source" in domain.lower():
        github_hint = (
            "\n\nGỢI Ý CHỦ ĐỀ GITHUB/OPEN-SOURCE (ưu tiên):"
            "\n- Phân tích kỹ thuật của một repo GitHub nổi tiếng (ví dụ: llm.c của Karpathy,"
            " whisper, stable-diffusion, transformers, langchain, vllm, llama.cpp, comfyui...)"
            "\n- Giải thích cách hoạt động bên trong của một open-source tool phổ biến"
            "\n- Cách đọc và hiểu source code của một dự án ML/AI nổi tiếng"
            "\nCHÚ Ý: Tập trung vào MỘT repo cụ thể, phân tích đến tầng thuật toán/kiến trúc."
        )
    elif "thuật toán ai" in domain.lower() or "ml" in domain.lower():
        github_hint = (
            "\n\nGỢI Ý CHỦ ĐỀ THUẬT TOÁN AI/ML (ưu tiên):"
            "\n- Cơ chế toán học đằng sau: attention/transformer, backprop, diffusion, RLHF"
            "\n- Tại sao thuật toán X hoạt động và tại sao cách truyền thống thất bại"
            "\n- Trace từng bước tính toán với ma trận/tensor cụ thể (không chỉ công thức)"
        )

    prompt = f"""Ban la bien tap vien kenh YouTube ve cong nghe theo phong cach giai thich truc quan (visualization).
Hay de xuat MOT chu de video cu the, hap dan trong linh vuc: {domain}.

Yeu cau:
- Chu de du hep de giai thich trong video 5 phut bang hinh anh dong/bieu do.
- Phu hop de minh hoa bang animation, bieu do du lieu, hoac mo phong thuat toan.
- Viet bang {lang_name}.
- KHONG trung hoac qua giong cac chu de da dung duoi day:
{used_block}{trend_section}{github_hint}

Chi tra ve DUY NHAT ten chu de tren mot dong, khong giai thich, khong danh so, khong dau ngoac kep."""

    topic = generate_topic(prompt)
    log.info("Chu de da chon: %s", topic)
    return topic


def generate_topic(prompt: str) -> str:
    """Gọi LLM lấy chủ đề. Tách riêng để dễ test."""
    from .llm import generate
    return generate(prompt, task="topic").strip().splitlines()[0].strip().strip('"').strip("-").strip()
