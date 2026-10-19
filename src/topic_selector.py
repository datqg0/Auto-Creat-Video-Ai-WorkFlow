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
        if any(k in domain.lower() for k in ("graphic programming", "shader", "ray tracing", "render")):
            github_hint = (
                "\n\nGRAPHIC PROGRAMMING & SHADERS HINTS (priority):"
                "\n- Visualize how code turns into photons and pixels: Ray Tracing, Pixel Shaders, Signed Distance Functions (SDF), Raymarching, Voxel engines, Rasterization pipeline."
                "\n- Example: 'Inside Ray Tracing: How Computers Bounce 1 Billion Photons Per Second', 'Why Mirrors Were 3D Gaming's Worst Nightmare', 'The Math Behind Shaders'."
            )
        elif any(k in domain.lower() for k in ("algorithm", "logic", "pathfinding", "sorting", "boids")):
            github_hint = (
                "\n\nVISUAL ALGORITHMS & LOGIC HINTS (priority):"
                "\n- Step-by-step visual trace of classic algorithms: A* Pathfinding, Boids Flocking simulation, Quadtree spatial partitioning, Huffman compression, Quicksort/Merge sort."
                "\n- Example: 'A* Algorithm: How Game Characters Always Find the Shortest Path', 'Boids: 3 Simple Rules Creating Natural Flocking Intelligence'."
            )
        elif any(k in domain.lower() for k in ("hardware", "silicon", "transistor", "cpu", "gpu", "cache", "asml", "ssd")):
            github_hint = (
                "\n\nHARDWARE & SILICON WONDERS HINTS (priority):"
                "\n- Physical marvels of silicon and computing architecture: 3nm Transistors, CPU Cache L1/L2/L3 bottlenecks, CPU vs GPU architecture, ASML EUV lithography machines, Flash SSD vs HDD."
                "\n- Example: 'Inside a 3nm Chip: Trapping 50 Billion Transistors on a Fingernail', 'Why Your PC Would Run 100x Slower Without CPU Cache'."
            )
        elif any(k in domain.lower() for k in ("cryptography", "security", "rsa", "diffie", "hash", "steganography")):
            github_hint = (
                "\n\nCRYPTOGRAPHY & DIGITAL SECURITY HINTS (priority):"
                "\n- Mathematical paradoxes protecting digital secrets: Diffie-Hellman color-mixing key exchange, RSA prime factor trapdoor, SHA-256 one-way hashing, Zero-Knowledge Proofs, Steganography."
                "\n- Example: 'Diffie-Hellman: Sharing a Secret Key in Plain Sight', 'The Giant Prime Numbers Protecting All Global Banking (RSA)'."
            )
        elif any(k in domain.lower() for k in ("graphics", "rendering", "bézier", "bezier", "anti-aliasing", "pbr", "culling")):
            github_hint = (
                "\n\nCOMPUTER GRAPHICS & VISUAL COMPUTING HINTS (priority):"
                "\n- Geometric and optical tricks that fool human vision: Bézier curves, Anti-Aliasing (MSAA/DLSS), Physically Based Rendering (PBR), Occlusion Culling, Shadow Mapping."
                "\n- Example: 'The Magic Bézier Curve: 4 Points That Shaped Every Digital Font', 'How Games Run Smoothly by Not Drawing What You Cannot See'."
            )
        elif any(k in domain.lower() for k in ("math", "art of math", "fractal", "mandelbrot", "conway", "fourier", "voronoi", "chaos")):
            github_hint = (
                "\n\nART OF MATH & SIMULATION HINTS (priority):"
                "\n- Mind-bending visual beauty of pure math: Mandelbrot infinite fractals, Conway's Game of Life emergence, Fourier transform drawing any shape with rotating epicycles, Voronoi diagrams, Chaos theory."
                "\n- Example: 'The Mandelbrot Set: An Infinite Universe Inside a Simple Formula', 'Conway's Game of Life: How 4 Rules Spawn Digital Organisms'."
            )
        elif any(k in domain.lower() for k in ("everyday tech", "lifestyle", "gadget", "qr", "cable", "battery", "wifi", "gps", "incognito")):
            github_hint = (
                "\n\nEVERYDAY TECH & INTERNET MYSTERIES HINTS (priority):"
                "\n- Revealing the hidden tech inside objects we use every day: QR Code Reed-Solomon error correction, Subsea fiber-optic cables, GPS and Einstein's Relativity, Lithium battery degradation, Incognito mode."
                "\n- Example: 'The Secret of QR Codes: Why 30% Damage Won't Stop Them Scanning', 'How Your Text Travels Under the Ocean to the US in 0.1s'."
            )
        elif any(k in domain.lower() for k in ("ai", "machine learning", "ml", "attention", "diffusion", "neural")):
            github_hint = (
                "\n\nAI & MACHINE LEARNING VISUALIZED HINTS (priority):"
                "\n- Deep visual explanation of modern AI mechanisms: Attention Mechanism in Transformers, Diffusion process turning noise into art, Vector Embeddings, Gradient Descent loss landscape."
                "\n- Example: 'Inside Attention Mechanism: How ChatGPT Actually Understands Context', 'How Diffusion Models Create Art Out of Pure Noise'."
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
        if any(k in domain.lower() for k in ("lập trình đồ họa", "graphic programming", "shader", "ray tracing", "raymarching", "rasterization")):
            github_hint = (
                "\n\nGỢI Ý CHỦ ĐỀ LẬP TRÌNH ĐỒ HỌA & SHADERS (GRAPHIC PROGRAMMING - THỊ GIÁC ĐỈNH CAO):"
                "\n- Cốt lõi: Bóc tách cách từng dòng code toán học biến thành hàng triệu điểm ảnh lộng lẫy trên màn hình."
                "\n  + 'Ray Tracing: Làm sao máy tính tính toán quỹ đạo 1 tỷ tia sáng mỗi giây trong bóng tối?'"
                "\n  + 'Tại sao gương phản chiếu và mặt nước từng là cơn ác mộng lớn nhất của lập trình viên game 3D?'"
                "\n  + 'Pixel Shaders: Bí thuật giúp một phương trình toán học vẽ nên cả một đại dương sóng vỗ'"
                "\n  + 'Raymarching & Signed Distance Functions: Cách vẽ vật thể 3D siêu thực mà không cần một tam giác nào'"
                "\n  + 'Voxel Engine: Giải mã thuật toán biến các khối lập phương thành thế giới sống động như Minecraft'"
                "\n  + 'Rasterization: Cỗ máy biến các đa giác 3D thành điểm ảnh màn hình trong 0.001 giây hoạt động ra sao?'"
                "\n- Phong cách: Đặt câu hỏi tò mò mạnh, kích thích trực quan thị giác, kết hợp Manim và HTML5 Motion Graphics."
            )
        elif any(k in domain.lower() for k in ("thuật toán trực quan", "algorithms", "tư duy logic", "tìm đường", "boids", "sắp xếp")):
            github_hint = (
                "\n\nGỢI Ý CHỦ ĐỀ THUẬT TOÁN TRỰC QUAN & TƯ DUY LOGIC (ALGORITHMS - 3BLUE1BROWN STYLE):"
                "\n- Cốt lõi: Lần theo từng bước chuyển động (step-by-step trace) của dữ liệu, biến thuật toán trừu tượng thành điệu nhảy thị giác."
                "\n  + 'Thuật toán A*: Làm sao nhân vật trong game luôn tìm ra đường đi ngắn nhất qua mê cung hiểm trở?'"
                "\n  + 'Boids: Chỉ với 3 quy tắc cực đơn giản, làm sao hàng vạn con chim ảo bay lượn y hệt tự nhiên?'"
                "\n  + 'Cây Quadtree & Phân vùng không gian: Bí mật giúp game xử lý va chạm 10.000 quái vật mà không tụt FPS'"
                "\n  + 'Thuật toán nén Huffman: Làm sao nén cuốn tiểu thuyết dày 1.000 trang thành chuỗi nhị phân tí hon?'"
                "\n  + 'Quicksort vs Merge sort: Cuộc đối đầu tốc độ giữa hai triết lý chia để trị vĩ đại nhất lịch sử'"
                "\n  + 'Thuật toán Ant Colony: Tại sao đàn kiến mù lại dạy các kỹ sư cách tối ưu hóa mạng viễn thông toàn cầu?'"
            )
        elif any(k in domain.lower() for k in ("phần cứng", "hardware", "bán dẫn", "transistor", "cpu", "gpu", "cache", "asml", "ssd")):
            github_hint = (
                "\n\nGỢI Ý CHỦ ĐỀ PHẦN CỨNG MÁY TÍNH & BÁN DẪN (HARDWARE - KỲ QUAN CÔNG NGHỆ):"
                "\n- Cốt lõi: Bóc tách kỳ quan vật lý và kỹ thuật điện tử thu nhỏ đến cấp độ nguyên tử."
                "\n  + 'Bên trong chip 3nm: Con người làm thế nào nhốt được 50 tỷ bóng bán dẫn vào móng tay?'"
                "\n  + 'Tại sao máy tính của bạn sẽ chậm đi 100 lần nếu nhà sản xuất cắt bỏ bộ nhớ Cache L1/L2/L3?'"
                "\n  + 'CPU vs GPU: Tại sao 1 bộ não thiên tài lại thua 10.000 học sinh tiểu học khi vẽ đồ họa 3D?'"
                "\n  + 'Cỗ máy quang khắc EUV của ASML: Cỗ máy phức tạp nhất hành tinh bắn giọt thiếc nóng chảy tạo tia laser ra sao?'"
                "\n  + 'Ổ SSD Flash vs Đĩa từ HDD: Làm sao nhốt hàng tỷ electron trong bẫy điện tích suốt 10 năm không mất điện?'"
                "\n  + 'Cổng logic số: Làm thế nào từ những chiếc công tắc tắt/bật 0-1 lại sinh ra được trí thông minh nhân tạo?'"
            )
        elif any(k in domain.lower() for k in ("mã hóa", "cryptography", "an toàn số", "rsa", "diffie", "hash", "bảo mật", "steganography")):
            github_hint = (
                "\n\nGỢI Ý CHỦ ĐỀ MÃ HÓA & AN TOÀN SỐ (CRYPTOGRAPHY - BẢO VỆ THẾ GIỚI SỐ):"
                "\n- Cốt lõi: Những nghịch lý toán học kỳ diệu giúp con người giữ bí mật tuyệt đối giữa thanh thiên bạch nhật."
                "\n  + 'Trao đổi khóa Diffie-Hellman: Nghịch lý pha màu giúp 2 người gửi chìa khóa mà không ai cướp được'"
                "\n  + 'Bí mật số nguyên tố RSA: Phép toán nhân thì siêu dễ nhưng tính ngược lại mất cả triệu năm của vũ trụ'"
                "\n  + 'Hàm băm SHA-256: Tại sao cỗ máy xay sinh tố dữ liệu này lại là xương sống bảo vệ toàn bộ Internet?'"
                "\n  + 'Zero-Knowledge Proofs: Làm sao chứng minh bạn biết mật mã kho báu mà không cần nói mật mã ra?'"
                "\n  + 'Giấu tin Steganography: Kỹ thuật giấu một bức thư tuyệt mật bên trong từng pixel của tấm ảnh chú mèo'"
                "\n  + 'Ổ khóa xanh HTTPS: Trình duyệt của bạn bắt tay với máy chủ Google an toàn như thế nào trong 0.05 giây?'"
            )
        elif any(k in domain.lower() for k in ("đồ họa máy tính", "thị giác", "bézier", "bezier", "anti-aliasing", "pbr", "culling", "shadow mapping")):
            github_hint = (
                "\n\nGỢI Ý CHỦ ĐỀ ĐỒ HỌA MÁY TÍNH & THỊ GIÁC (COMPUTER GRAPHICS & RENDERING):"
                "\n- Cốt lõi: Những mẹo toán học và đánh lừa thị giác đỉnh cao giúp mắt người tin vào thế giới ảo."
                "\n  + 'Phép màu đường cong Bézier: 4 điểm neo toán học đã định hình toàn bộ font chữ và đồ họa vector ra sao?'"
                "\n  + 'Khử răng cưa Anti-Aliasing: Tại sao các cạnh bậc thang thô ráp lại biến mất mịn màng trên màn hình?'"
                "\n  + 'Occlusion Culling: Mẹo ảo thuật giúp game 3D thế giới mở chạy mượt mà bằng cách... không vẽ những gì bạn không thấy'"
                "\n  + 'Chiếu sáng vật lý PBR: Làm sao máy tính phân biệt được ánh kim loại phản chiếu khác lớp nhựa bóng?'"
                "\n  + 'Không gian màu RGB vs CMYK vs HDR: Tại sao màu sắc bạn thấy trên màn hình in ra giấy lại khác hoàn toàn?'"
                "\n  + 'Shadow Mapping: Cách máy tính nhìn cảnh vật từ góc nhìn của ngọn đèn để tạo ra bóng đổ chân thực'"
            )
        elif any(k in domain.lower() for k in ("nghệ thuật của toán học", "art of math", "fractal", "mandelbrot", "conway", "fourier", "voronoi", "hỗn loạn", "tôpô")):
            github_hint = (
                "\n\nGỢI Ý CHỦ ĐỀ NGHỆ THUẬT CỦA TOÁN HỌC & MÔ PHỎNG (ART OF MATH - VẺ ĐẸP VÔ TẬN):"
                "\n- Cốt lõi: Vẻ đẹp thị giác choáng ngợp của các quy luật toán học thuần túy và trật tự trỗi dậy từ hỗn loạn."
                "\n  + 'Tập hợp Mandelbrot: Phóng to vô tận vẫn thấy các vũ trụ hình học mới từ công thức z = z² + c'"
                "\n  + 'Trò chơi cuộc sống Conway: Chỉ với 4 luật cực đơn giản, các tế bào số tự sinh sôi và tiến hóa như sinh vật sống'"
                "\n  + 'Biến đổi Fourier: Vòng tròn quay kỳ diệu có thể vẽ nên bất kỳ bức tranh hay phân tích mọi âm thanh thế giới'"
                "\n  + 'Biểu đồ Voronoi: Quy luật phân chia không gian kỳ diệu từ cánh chuồn chuồn đến trạm thu phát sóng di động'"
                "\n  + 'Thuyết hỗn loạn & Hiệu ứng cánh bướm: Tại sao chỉ một sai số phần tỷ lại làm dự báo thời tiết sai hoàn toàn sau 7 ngày?'"
                "\n  + 'Hình học Tôpô: Tại sao trong mắt nhà toán học, một chiếc cốc cà phê và một chiếc bánh donut là cùng một thứ?'"
            )
        elif any(k in domain.lower() for k in ("đời sống", "thiết bị", "lifestyle", "gadget", "qr", "cáp quang", "pin", "wifi", "gps", "sạc", "nén", "jpeg", "mp3")):
            github_hint = (
                "\n\nGỢI Ý CHỦ ĐỀ CÔNG NGHỆ ĐỜI SỐNG & THIẾT BỊ XUNG QUANH (EVERYDAY TECH - ĐẠI CHÚNG 100%):"
                "\n- Cốt lõi: Giải mã những thứ người bình thường dùng mỗi ngày nhưng không hề biết phép màu công nghệ bên trong."
                "\n  + 'Bí mật mã QR: Tại sao bị rách góc hay vẽ bẩn 30% máy vẫn quét ra bình thường (thuật toán Reed-Solomon)?'"
                "\n  + 'Cáp quang biển xuyên đại dương: Tin nhắn Alo của bạn chui xuống đáy biển 4.000m đến Mỹ trong 0.1s ra sao?'"
                "\n  + 'Định vị GPS: Tại sao nếu không có Thuyết tương đối của Einstein, định vị trên điện thoại sẽ sai 11km mỗi ngày?'"
                "\n  + 'Hiện tượng chai pin Lithium: Chuyện gì thực sự xảy ra với các ion pin sau 2 năm sạc liên tục?'"
                "\n  + 'Chế độ ẩn danh Incognito: Những gì nó thực sự giấu được và những gì nhà mạng/Google vẫn theo dõi?'"
                "\n  + 'Sóng Wifi vs Lò vi sóng: Tại sao bật lò vi sóng lại làm chập chờn mạng Internet trong nhà?'"
                "\n  + 'Nén ảnh JPEG / MP3: Máy tính đã đánh lừa mắt và tai của chúng ta tinh vi như thế nào?'"
                "\n  + 'Nhiếp ảnh điện toán: Tại sao camera điện thoại bé tí lại chụp đêm sáng rõ hơn cả mắt người?'"
            )
        elif any(k in domain.lower() for k in ("trí tuệ nhân tạo", "ai", "machine learning", "ml", "attention", "diffusion", "nơ-ron")):
            github_hint = (
                "\n\nGỢI Ý CHỦ ĐỀ TRÍ TUỆ NHÂN TẠO & HỌC MÁY TRỰC QUAN (AI VISUALIZED):"
                "\n- Cốt lõi: Mổ xẻ cơ chế toán học và hình học đằng sau các mô hình AI đột phá nhất hiện nay."
                "\n  + 'Attention Mechanism: Ma trận chú ý thực sự giúp ChatGPT hiểu được ngữ cảnh câu nói con người ra sao?'"
                "\n  + 'Mô hình Diffusion: Làm sao AI vẽ nên bức tranh tuyệt mỹ chỉ từ một màn hình đầy nhiễu hạt tuyết?'"
                "\n  + 'Vector Embeddings: Làm thế nào máy tính đo được khoảng cách ý nghĩa giữa từ Vua, Nữ hoàng và Vương miện?'"
                "\n  + 'Gradient Descent: Trực giác hình học về quả bóng lăn xuống thung lũng giúp AI tự học từ sai lầm'"
                "\n  + 'Mạng nơ-ron CNN: Thị giác máy tính bóc tách từng đường nét để nhận diện khuôn mặt con người như thế nào?'"
            )

        prompt = f"""Bạn là biên tập viên kênh YouTube về công nghệ theo phong cách giải thích trực quan (visualization).
Hãy đề xuất MỘT chủ đề video cụ thể, hấp dẫn trong lĩnh vực: {domain}.

Yêu cầu:
- Chủ đề đủ hẹp để giải thích trong video 5 phút bằng hình ảnh động/biểu đồ.
- Phù hợp để minh họa bằng animation, biểu đồ dữ liệu, hoặc mô phỏng thuật toán.
- Đặt tên chủ đề theo phong cách hấp dẫn, khơi gợi tò mò mạnh (như "Hóa ra...", "Tại sao... lại... đến vậy?", "Không ngờ...", hoặc nghịch lý con số). TUYỆT ĐỐI không dùng văn học thuật khô cứng như "Tổng quan về...", "Tìm hiểu...", "Cơ chế...".
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
