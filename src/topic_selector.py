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

    if lang != "vi":
        trend_section = (
            f"\n\nCURRENT TECH TRENDS (use to pick a timely and relevant topic):\n{trend_block}\n"
            "Please refer to these trends to pick a topic people currently care about,"
            " but DO NOT copy the article title verbatim. Adapt it into a visual explanation perspective."
            if trend_block else ""
        )

        github_hint = ""
        if any(k in domain.lower() for k in ("github", "open-source", "repo")):
            github_hint = (
                "\n\nGITHUB/OPEN-SOURCE TOPIC HINTS (priority):"
                "\n- Technical deep-dive into a specific GitHub repo (e.g., Karpathy's llm.c,"
                " llama.cpp, vllm, BitNet, Ollama, ComfyUI, whisper.cpp, open-webui, airllm, deepseek-v3...)"
                "\n- REQUIREMENT: Explicitly mention the specific repo name in the title"
                " (e.g., 'Inside llama.cpp: Running LLMs on Pure CPU', 'Microsoft BitNet Explained: The 1-Bit LLM Architecture')"
                "\n- Analyze the core architectural breakthrough or algorithmic trick inside the repo."
            )
        elif any(k in domain.lower() for k in ("thuật toán ai", "thuat toan ai", "algorithm", "ml", "ai")):
            github_hint = (
                "\n\nAI/ML ALGORITHM TOPIC HINTS (priority):"
                "\n- Mathematical mechanism behind: attention/transformer, backprop, diffusion, RLHF"
                "\n- Why algorithm X works and why traditional approaches fail"
                "\n- Step-by-step trace of computations with concrete matrices/tensors (not just formulas)"
            )
        elif any(k in domain.lower() for k in ("toán", "math", "đại số", "algebra", "giải tích", "calculus", "xác suất", "probability", "fourier")):
            github_hint = (
                "\n\nFOUNDATIONAL MATH VISUALIZATION HINTS (3Blue1Brown style, priority):"
                "\n- Geometric intuition behind classic math: Linear transformations, Determinants, Eigenvectors, Derivative as slope/zoom, Integral as accumulation, Bayes' Theorem, Fourier Transform, Complex numbers rotation."
                "\n- Frame as an intuitive visual revelation answering: 'What is the true geometric meaning of X?'"
            )
        elif any(k in domain.lower() for k in ("căn bản", "can ban", "fundamentals", "basics", "mạng", "bộ nhớ", "hệ điều hành", "cấu trúc dữ liệu")):
            github_hint = (
                "\n\nCS/IT FUNDAMENTALS HINTS (priority):"
                "\n- Explain classic, fundamental IT concepts every developer needs to know (Stack vs Heap, Pointer, TCP 3-way handshake, Process vs Thread, Hash Table collision, DNS resolution...)"
                "\n- Frame as an intuitive, visual walkthrough solving a clear problem."
            )

        prompt = f"""You are an editor for a tech YouTube channel specializing in visual explanations (visualization).
Propose ONE specific, engaging video topic in the field: {domain}.

Requirements:
- Narrow enough to explain in a 5-minute video using animations/diagrams.
- Well-suited for illustration with animations, data charts, or algorithm simulations.
- Written in English.
- Do NOT duplicate or closely resemble any of the previously used topics below:
{used_block}{trend_section}{github_hint}

Return ONLY the topic title on a single line, with no explanations, no numbering, and no quotes."""
    else:
        trend_section = (
            f"\n\nXU HƯỚNG CÔNG NGHỆ HIỆN TẠI (sử dụng để chọn chủ đề phù hợp thời sự):\n{trend_block}\n"
            "Hãy THAM KHẢO các xu hướng này để chọn chủ đề gần với điều người dùng đang quan tâm,"
            " nhưng KHÔNG chép nguyên tiêu đề bài viết. Biến đổi thành góc nhìn giải thích trực quan."
            if trend_block else ""
        )

        github_hint = ""
        if any(k in domain.lower() for k in ("github", "open-source", "repo")):
            github_hint = (
                "\n\nGỢI Ý CHỦ ĐỀ GITHUB/OPEN-SOURCE (ưu tiên):"
                "\n- Phân tích kỹ thuật của một repo GitHub cụ thể (ví dụ: llama.cpp, vLLM, llm.c của Karpathy,"
                " BitNet, Ollama, ComfyUI, whisper.cpp, open-webui, airllm, deepseek-v3...)"
                "\n- BẮT BUỘC: Đưa tên Repo cụ thể vào tiêu đề video (ví dụ: 'Bên trong repo llama.cpp: Bí quyết chạy LLM không cần GPU',"
                " 'Giải mã repo BitNet của Microsoft: Kỷ nguyên LLM 1-bit...')"
                "\n- Đi sâu vào kiến trúc code, thuật toán hoặc kỹ thuật tối ưu cốt lõi làm nên sức mạnh của repo đó."
            )
        elif any(k in domain.lower() for k in ("toán", "math", "đại số", "algebra", "giải tích", "calculus", "xác suất", "probability", "fourier")):
            github_hint = (
                "\n\nGỢI Ý CHỦ ĐỀ TOÁN HỌC CĂN BẢN (Phong cách 3Blue1Brown & Manim trực quan):"
                "\n- Khai thác trực quan hình học đằng sau các khái niệm toán học kinh điển, biến công thức trừu tượng thành chuyển động trực giác:"
                "\n  + 'Bản chất hình học của Ma trận: Phép kéo giãn và biến hình không gian'"
                "\n  + 'Định thức (Determinant) là gì: Hệ số co giãn diện tích mà SGK chưa nói'"
                "\n  + 'Eigenvector & Eigenvalue: Những chiếc trục bất động khi không gian bị kéo giãn'"
                "\n  + 'Đạo hàm dưới kính lúp hình học: Phóng to đến vô cùng để tìm độ nghiêng tức thời'"
                "\n  + 'Định lý Bayes: Công thức toán học của niềm tin và trực giác con người'"
                "\n  + 'Biến đổi Fourier: Chiếc lăng kính chia tách mọi âm thanh thành sóng sin'"
                "\n  + 'Số ảo i và công thức Euler: Chiếc chìa khóa xoay tròn 90 độ trong không gian 2D'"
                "\n- Phong cách: Trả lời câu hỏi 'Bản chất thực sự của nó là gì?', cực kỳ phù hợp để vẽ Manim/LaTeX động."
            )
        elif any(k in domain.lower() for k in ("thuật toán ai", "thuat toan ai", "algorithm", "ml", "ai")):
            github_hint = (
                "\n\nGỢI Ý CHỦ ĐỀ THUẬT TOÁN AI/ML (ưu tiên):"
                "\n- Cơ chế toán học đằng sau: attention/transformer, backprop, diffusion, RLHF"
                "\n- Tại sao thuật toán X hoạt động và tại sao cách truyền thống thất bại"
                "\n- Trace từng bước tính toán với ma trận/tensor cụ thể (không chỉ công thức)"
            )
        elif any(k in domain.lower() for k in ("căn bản", "can ban", "fundamentals", "mạng", "bộ nhớ", "hệ điều hành", "cấu trúc dữ liệu", "web")):
            github_hint = (
                "\n\nGỢI Ý CHỦ ĐỀ KIẾN THỨC IT CĂN BẢN (Nền tảng Khoa học máy tính):"
                "\n- Giải thích những khái niệm kinh điển mà mọi lập trình viên/dân IT đều cần hiểu sâu sắc nhưng thường thấy mơ hồ:"
                "\n  + 'Stack vs Heap: Bản đồ vùng nhớ máy tính hoạt động thế nào?'"
                "\n  + 'Bắt tay 3 bước TCP: Bí quyết Internet truyền dữ liệu không rơi rớt'"
                "\n  + 'Con trỏ (Pointer) trong bộ nhớ: Biến số thực sự nằm ở đâu trong RAM?'"
                "\n  + 'Process vs Thread: Đa nhiệm trên CPU thực sự diễn ra thế nào?'"
                "\n  + 'Hash Table: Cách tra cứu O(1) và giải quyết đụng độ băm'"
                "\n  + 'DNS hoạt động ra sao: 4 trạm dừng biến tên miền thành địa chỉ IP'"
                "\n- Phong cách: Đặt câu hỏi kích thích tò mò, trực quan hóa đường đi của dữ liệu, ô nhớ RAM và luồng tín hiệu."
            )

        prompt = f"""Bạn là biên tập viên kênh YouTube về công nghệ theo phong cách giải thích trực quan (visualization).
Hãy đề xuất MỘT chủ đề video cụ thể, hấp dẫn trong lĩnh vực: {domain}.

Yêu cầu:
- Chủ đề đủ hẹp để giải thích trong video 5 phút bằng hình ảnh động/biểu đồ.
- Phù hợp để minh họa bằng animation, biểu đồ dữ liệu, hoặc mô phỏng thuật toán.
- Viết bằng tiếng Việt.
- KHÔNG trùng hoặc quá giống các chủ đề đã dùng dưới đây:
{used_block}{trend_section}{github_hint}

Chỉ trả về DUY NHẤT tên chủ đề trên một dòng, không giải thích, không đánh số, không dấu ngoặc kép."""

    topic = generate_topic(prompt)
    log.info("Chu de da chon: %s", topic)
    return topic


def generate_topic(prompt: str) -> str:
    """Gọi LLM lấy chủ đề. Tách riêng để dễ test."""
    from .llm import generate
    return generate(prompt, task="topic").strip().splitlines()[0].strip().strip('"').strip("-").strip()
