"""Series Manager: Lên kế hoạch và điều phối chuỗi video theo chủ đề (Series Engine).

Tính năng:
  - Tự động phát hiện khi DB trống hoặc series cũ đã hoàn thành -> Gọi AI lập giáo trình chuỗi mới.
  - Phân rã chủ đề lớn thành lộ trình 3-7 tập có tính sư phạm kết nối chặt chẽ (scaffolding).
  - Tự động sinh hook mở đầu (kế thừa tập trước) và hook kết thúc (gợi mở tập sau).
  - Cung cấp context cho kịch bản và theo dõi tiến độ từng tập.
"""
from __future__ import annotations

import json
import logging
import random
import re
from typing import Any

from . import db
from .config import CONFIG
from .llm import generate

log = logging.getLogger(__name__)


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]
    return json.loads(text)


def plan_new_series(
    theme: str | None = None,
    domain: str | None = None,
    num_episodes: int = 5,
    mode: str = "long",
) -> dict[str, Any]:
    """Sử dụng LLM để lên giáo trình lộ trình cho một chuỗi video mới (tách biệt rõ Short vs Long) và lưu vào DB."""
    db.init_db()

    normalized_mode = "short" if mode == "short" else ("mega" if mode == "mega" else "long")
    is_short = normalized_mode == "short"

    topics_cfg = CONFIG.get("topics", {})
    domains: list[str] = topics_cfg.get(
        "domains",
        [
            "Lập trình đồ họa & Shaders (Graphic Programming: Ray Tracing, Shaders, Voxels, Render Pipeline)",
            "Thuật toán trực quan & Tư duy logic (Algorithms: A*, Boids, Quadtree, Nén dữ liệu, Sắp xếp)",
            "Phần cứng máy tính & Bán dẫn (Hardware: Transistor, Cổng logic, Cache L1/L2/L3, CPU vs GPU, ASML)",
            "Mã hóa & An toàn số (Cryptography: Diffie-Hellman, RSA, Hàm băm SHA, Zero-Knowledge Proofs)",
            "Đồ họa máy tính & Thị giác (Computer Graphics: Bézier, Anti-Aliasing, PBR, Occlusion Culling)",
            "Nghệ thuật của Toán học & Mô phỏng (Art of Math: Mandelbrot, Conway's Game of Life, Fourier, Voronoi)",
            "Công nghệ đời sống & Internet căn bản (Everyday Tech: Mã QR, Cáp quang biển, GPS, Wifi, Pin Lithium)",
            "Trí tuệ nhân tạo & Học máy trực quan (AI & Machine Learning: Attention, Diffusion, CNN, Gradient Descent)",
            "Ngôn ngữ lập trình & Cuộc chiến hiệu năng (Programming Languages: Rust vs C++, Python vs C, V8 JIT, Go vs Java)",
        ],
    )
    if is_short:
        sel_domain = domain or "Thuật toán trực quan & Kỳ quan toán học (Phong cách Shorts @onjsdev: Sorting Algorithms, Pathfinding & Mazes, Fast Pi Formulas, Graph Traversals)"
    else:
        sel_domain = domain or "Trí tuệ nhân tạo & Học máy từ số 0 (Deep-ML & AI from Scratch: Transformers, Attention, Neural Networks, CNN, Loss Functions, Gradient Descent, Backpropagation, Optimizers)"

    used_topics = db.recent_topics(days=60)
    used_block = "\n".join(f"- {t}" for t in used_topics[:25]) if used_topics else "(Chưa có chủ đề nào)"

    theme_prompt = f'về chủ đề: "{theme}"' if theme else f"trong lĩnh vực công nghệ: {sel_domain}"

    lang = CONFIG.get("language", "vi")
    lang_name = "tiếng Việt" if lang == "vi" else "English"

    if is_short:
        format_instructions = f"""YÊU CẦU ĐẶC BIỆT CHO CHUỖI YOUTUBE SHORTS (9:16, 60–180 GIÂY MỖI TẬP - PHONG CÁCH @ONJSDEV):
1. "series_name": Tên chuỗi ngắn gọn, giật gân, siêu kích thích tò mò bằng {lang_name} theo phong cách @onjsdev (Ví dụ: "60 Giây Giải Thuật Trực Quan", "Cuộc Đua Thuật Toán Sắp Xếp", "Kỳ Quan Mê Cung & Đồ Thị", "Bí Mật Toán Học Máy Tính").
2. Lộ trình {num_episodes} tập Shorts phải độc lập nhưng liên kết chặt chẽ (Series Hook):
   - Mỗi tập trực quan hóa TRỌN VẸN 1 thuật toán, 1 cấu trúc dữ liệu, hoặc 1 công thức toán học kỳ thú (Bubble Sort, QuickSort, Merge Sort, Dijkstra, A*, Mê cung Hunt and Kill, Công thức Pi Chudnovsky) có thể mô phỏng động trực quan trong 60 giây.
   - Tập 1: hook_from_previous để rỗng.
   - Từ tập 2 trở đi: hook_from_previous là 1 câu ngắn gợi lại tập trước (Ví dụ: "Ở short trước chúng ta đã thấy vì sao Bubble Sort lại chậm chạp...").
   - hook_to_next: 1 câu kích thích người xem đón chờ short tiếp theo (Ví dụ: "Nhưng thuật toán ở short sau còn nhanh hơn gấp 10 lần!")."""
    else:
        format_instructions = f"""YÊU CẦU ĐẶC BIỆT CHO CHUỖI VIDEO DÀI CHUYÊN AI (16:9, CHIỀU SÂU BẢN CHẤT 5–15 PHÚT - PHONG CÁCH DEEP-ML):
1. "series_name": Tên chuỗi video chuyên sâu AI & Deep Learning từ số 0 bằng {lang_name} (Ví dụ: "Xây Dựng Deep Learning Từ Số 0 Bằng NumPy", "Bí Mật Transformers & Attention Mechanism", "Toán Học Cốt Lõi Của Trí Tuệ Nhân Tạo", "Tự Lập Trình Mạng Nơ-ron Không Cần Framework").
2. Lộ trình {num_episodes} tập xây dựng từ nền tảng đến kiến trúc tối tân dựa trên các bài toán chuẩn của Deep-ML (deep-ml.com):
   - Tập 1: Trực giác Gradient Descent & Tự lập trình Hồi quy tuyến tính từ số 0
   - Tập 2: Logistic Regression & Hàm mất mát Cross-Entropy
   - Tập 3: Softmax Regression & Phân loại đa lớp
   - Tập 4: Mạng nơ-ron nhiều lớp (MLP) & Thuật toán lan truyền ngược (Backpropagation)
   - Tập 5: Lớp tích chập Convolution & Max Pooling (CNN từ số 0)
   - Tập 6: Cơ chế Masked Self-Attention & Scaled Dot-Product (Trái tim của Transformer & LLM)
   - Tập 7: Adam Optimizer & Kỹ thuật tối ưu hóa mô hình AI hiện đại
3. Hook kết nối:
   - hook_from_previous: Nhắc lại nền tảng từ tập trước để người xem nắm bắt mạch kiến thức.
   - hook_to_next: Gợi mở bài toán nan giải sẽ giải quyết ở tập tiếp theo."""

    prompt = f"""Bạn là Giám đốc Nội dung kênh YouTube công nghệ theo phong cách giải thích trực quan (visualization).
Hãy lên kế hoạch cho MỘT CHUỖI VIDEO (Series) gồm {num_episodes} tập {theme_prompt}.

{format_instructions}

3. KHÔNG trùng hoặc quá giống các chủ đề sau:
{used_block}

4. Mỗi tập gồm:
   - episode_num: Số thứ tự (1 đến {num_episodes})
   - topic: Tên chủ đề chi tiết của tập (đặt theo phong cách tò mò cao như kênh Kiến Thức Thú Vị kết hợp 3Blue1Brown: dùng "Hóa ra...", "Tại sao... lại... đến vậy?", "Không ngờ...", hoặc nghịch lý con số, TUYỆT ĐỐI tránh văn phong giáo trình khô cứng).
   - hook_from_previous: (Tập 1 để trống). Từ tập 2 trở đi, 1 câu ngắn nhắc lại nền tảng từ tập trước.
   - hook_to_next: (Tập cuối để trống). 1 câu gợi mở bí mật sẽ giải quyết ở tập tiếp theo.

Trả về DUY NHẤT một JSON hợp lệ dạng:
{{
  "series_name": "Tên chuỗi",
  "theme": "{theme or sel_domain}",
  "domain": "{sel_domain}",
  "description": "Mô tả ngắn 1-2 câu về chuỗi",
  "episodes": [
    {{
      "episode_num": 1,
      "topic": "Tên chủ đề tập 1",
      "hook_from_previous": "",
      "hook_to_next": "Câu gợi mở sang tập 2"
    }},
    ...
  ]
}}
Không kèm giải thích ngoài."""

    log.info("Đang nhờ AI thiết kế lộ trình Series [%s] %d tập (%s)...", normalized_mode.upper(), num_episodes, sel_domain)
    raw = generate(prompt, task="topic")
    data = _extract_json(raw)

    s_name = data.get("series_name") or f"Series {sel_domain.title()}"
    s_theme = data.get("theme") or (theme or sel_domain)
    eps_data = data.get("episodes", [])

    # Tránh trùng tên series
    existing = [s["name"] for s in db.list_all_series(mode=normalized_mode)]
    if s_name in existing:
        s_name = f"{s_name} (Phần {len(existing) + 1})"

    series_id = db.create_series(
        name=s_name,
        theme=s_theme,
        domain=sel_domain,
        total_episodes=len(eps_data) or num_episodes,
        mode=normalized_mode,
    )

    for ep in eps_data:
        db.add_series_episode(
            series_id=series_id,
            episode_num=int(ep.get("episode_num", 1)),
            topic=str(ep.get("topic", "")),
            hook_from_previous=str(ep.get("hook_from_previous", "")),
            hook_to_next=str(ep.get("hook_to_next", "")),
        )

    log.info("★ ĐÃ KHỞI TẠO SERIES MỚI [%s]: '%s' (%d tập) ★", normalized_mode.upper(), s_name, len(eps_data))
    for ep in eps_data:
        log.info("  [Tập %d/%d] %s", ep.get("episode_num"), len(eps_data), ep.get("topic"))

    series_dict = db.get_series_by_id(series_id) or {}
    series_dict["episodes"] = db.get_series_episodes(series_id)
    return series_dict


def get_or_create_active_series_task(mode: str = "long") -> dict[str, Any]:
    """Lấy tập video tiếp theo cần sản xuất theo đúng mode (long, short, mega).

    Quy tắc tự động:
      1. Tìm series đang 'in_progress' của riêng mode đó (Short series hoặc Long series riêng biệt).
      2. Nếu có series đang chạy:
         - Lấy tập 'pending' tiếp theo.
         - Nếu không còn tập pending nào -> Đánh dấu series là 'completed'.
      3. Nếu không có series nào đang chạy (DB trống hoặc series cũ vừa xong):
         - Tự động gọi plan_new_series(..., mode=mode) để tạo chuỗi video mới đúng định dạng!
         - Trả về tập 1 của series mới tạo.
    """
    db.init_db()
    normalized_mode = "short" if mode == "short" else ("mega" if mode == "mega" else "long")

    active = db.get_active_series(mode=normalized_mode)

    if active:
        next_ep = db.get_next_pending_episode(active["id"])
        if next_ep:
            log.info(
                "Tiếp tục chuỗi [%s] '%s' -> Tập %d/%d: %s",
                normalized_mode.upper(),
                active["name"],
                next_ep["episode_num"],
                active["total_episodes"],
                next_ep["topic"],
            )
            return {
                "series": active,
                "episode": next_ep,
                "series_context": {
                    "series_name": active["name"],
                    "episode_num": next_ep["episode_num"],
                    "total_episodes": active["total_episodes"],
                    "mode": normalized_mode,
                    "hook_from_previous": next_ep.get("hook_from_previous") or "",
                    "hook_to_next": next_ep.get("hook_to_next") or "",
                },
            }
        else:
            # Series đã hoàn thành tất cả các tập
            db.update_series(active["id"], status="completed")
            log.info("Series [%s] '%s' đã hoàn tất tất cả %d tập!", normalized_mode.upper(), active["name"], active["total_episodes"])

    # DB trống hoặc series vừa xong -> Tự động tạo series mới theo đúng mode!
    log.info("Database chưa có chuỗi [%s] đang chạy -> Tự động khởi tạo Chuỗi Video Mới (%s)...", normalized_mode.upper(), normalized_mode)
    s_cfg = CONFIG.get("series", {})
    if normalized_mode == "short":
        num_eps = int(s_cfg.get("short_episodes", s_cfg.get("default_episodes", 5)))
    else:
        num_eps = int(s_cfg.get("long_episodes", s_cfg.get("default_episodes", 5)))

    new_series = plan_new_series(num_episodes=num_eps, mode=normalized_mode)
    next_ep = db.get_next_pending_episode(new_series["id"])

    if not next_ep:
        raise RuntimeError(f"Khởi tạo series [{normalized_mode}] thất bại, không có tập nào được tạo.")

    return {
        "series": new_series,
        "episode": next_ep,
        "series_context": {
            "series_name": new_series["name"],
            "episode_num": next_ep["episode_num"],
            "total_episodes": new_series["total_episodes"],
            "mode": normalized_mode,
            "hook_from_previous": next_ep.get("hook_from_previous") or "",
            "hook_to_next": next_ep.get("hook_to_next") or "",
        },
    }


def print_series_table(mode: str | None = None) -> None:
    """Hiển thị bảng tiến độ các series trực quan trên terminal, phân loại rõ ràng Short vs Long."""
    db.init_db()
    all_series = db.list_all_series(mode=mode)

    if not all_series:
        print(f"\n[Series Engine] Hiện chưa có chuỗi video nào trong database (mode={mode or 'all'}).\n")
        return

    print("\n" + "=" * 78)
    print(" DANH SÁCH CÁC CHUỖI VIDEO (SERIES ROADMAP) TRONG DATABASE")
    print("=" * 78)

    status_icon = {
        "in_progress": "▶ ĐANG THỰC HIỆN",
        "completed": "✔ HOÀN TẤT",
    }
    ep_icon = {
        "pending": "[ ] Chưa làm",
        "in_progress": "[>] Đang làm",
        "rendered": "[✓] Đã render",
        "uploaded": "[★] Đã upload",
        "error": "[✗] Lỗi",
    }

    # Phân nhóm theo Short vs Long
    shorts_list = [s for s in all_series if s.get("mode") == "short"]
    longs_list = [s for s in all_series if s.get("mode") != "short"]

    def _render_group(title: str, series_items: list[dict[str, Any]]) -> None:
        if not series_items:
            return
        print(f"\n{title}")
        print("-" * 78)
        for s in series_items:
            st_text = status_icon.get(s["status"], s["status"])
            m_badge = f"[{s.get('mode', 'long').upper()}]"
            print(f"\n★ {m_badge} SERIES: {s['name']} ({st_text})")
            print(f"  Chủ đề: {s.get('theme', '')} | Chuyên ngành: {s.get('domain', '')} | Tiến độ: {s['completed_episodes']}/{s['total_episodes']} tập")
            print("  " + "-" * 74)

            for ep in s.get("episodes", []):
                ep_st = ep_icon.get(ep.get("status"), ep.get("status"))
                print(f"    Tập {ep['episode_num']:02d}: {ep['topic']:<45} {ep_st}")

    if shorts_list:
        _render_group("📱 CÁC CHUỖI VIDEO SHORTS (9:16 - ĐỘC LẬP)", shorts_list)

    if longs_list:
        _render_group("🖥️ CÁC CHUỖI VIDEO DÀI (16:9 - LONG / MEGA)", longs_list)

    print("\n" + "=" * 78 + "\n")
