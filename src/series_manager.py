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
) -> dict[str, Any]:
    """Sử dụng LLM để lên giáo trình lộ trình cho một chuỗi video mới và lưu vào DB."""
    db.init_db()

    topics_cfg = CONFIG.get("topics", {})
    domains: list[str] = topics_cfg.get("domains", ["algorithms", "architecture", "distributed_systems", "ai_llm"])
    sel_domain = domain or random.choice(domains)

    used_topics = db.recent_topics(days=60)
    used_block = "\n".join(f"- {t}" for t in used_topics[:25]) if used_topics else "(Chưa có chủ đề nào)"

    theme_prompt = f'về chủ đề: "{theme}"' if theme else f"trong lĩnh vực công nghệ: {sel_domain}"

    lang = CONFIG.get("language", "vi")
    lang_name = "tiếng Việt" if lang == "vi" else "English"

    prompt = f"""Bạn là Giám đốc Nội dung kênh YouTube công nghệ theo phong cách giải thích trực quan (visualization).
Hãy lên kế hoạch cho MỘT CHUỖI VIDEO (Series) gồm {num_episodes} tập {theme_prompt}.

YÊU CẦU NỘI DUNG:
1. "series_name": Tên chuỗi video ngắn gọn, ấn tượng, gây tò mò, viết bằng {lang_name} (Ví dụ: "Bí Mật Của Hệ Thống Phân Tán", "Kiến Trúc Transformer Từ A-Z").
2. Lộ trình {num_episodes} tập phải có tính sư phạm kết nối chặt chẽ (scaffolding):
   - Tập 1: Bản chất nguyên lý cốt lõi, dễ hiểu nhất, đặt ra bài toán lớn.
   - Các tập kế tiếp: Đi sâu vào từng cơ chế kỹ thuật, tăng dần độ sâu.
   - Tập cuối: Thực chiến quy mô lớn, tối ưu hóa hoặc tương lai.
3. KHÔNG trùng hoặc quá giống các chủ đề sau:
{used_block}

4. Mỗi tập gồm:
   - episode_num: Số thứ tự (1 đến {num_episodes})
   - topic: Tên chủ đề chi tiết của tập (cụ thể, súc tích)
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

    log.info("Đang nhờ AI thiết kế lộ trình Series %d tập (%s)...", num_episodes, sel_domain)
    raw = generate(prompt, task="topic")
    data = _extract_json(raw)

    s_name = data.get("series_name") or f"Series {sel_domain.title()}"
    s_theme = data.get("theme") or (theme or sel_domain)
    eps_data = data.get("episodes", [])

    # Tránh trùng tên series
    existing = [s["name"] for s in db.list_all_series()]
    if s_name in existing:
        s_name = f"{s_name} (Phần {len(existing) + 1})"

    series_id = db.create_series(
        name=s_name,
        theme=s_theme,
        domain=sel_domain,
        total_episodes=len(eps_data) or num_episodes,
    )

    for ep in eps_data:
        db.add_series_episode(
            series_id=series_id,
            episode_num=int(ep.get("episode_num", 1)),
            topic=str(ep.get("topic", "")),
            hook_from_previous=str(ep.get("hook_from_previous", "")),
            hook_to_next=str(ep.get("hook_to_next", "")),
        )

    log.info("★ ĐÃ KHỞI TẠO SERIES MỚI: '%s' (%d tập) ★", s_name, len(eps_data))
    for ep in eps_data:
        log.info("  [Tập %d/%d] %s", ep.get("episode_num"), len(eps_data), ep.get("topic"))

    series_dict = db.get_series_by_id(series_id) or {}
    series_dict["episodes"] = db.get_series_episodes(series_id)
    return series_dict


def get_or_create_active_series_task() -> dict[str, Any]:
    """Lấy tập video tiếp theo cần sản xuất.

    Quy tắc tự động:
      1. Tìm series đang 'in_progress'.
      2. Nếu có series đang chạy:
         - Lấy tập 'pending' tiếp theo.
         - Nếu không còn tập pending nào -> Đánh dấu series là 'completed'.
      3. Nếu không có series nào đang chạy (DB trống hoặc series cũ vừa xong):
         - Tự động gọi plan_new_series() để tạo chuỗi video mới!
         - Trả về tập 1 của series mới tạo.
    """
    db.init_db()

    active = db.get_active_series()

    if active:
        next_ep = db.get_next_pending_episode(active["id"])
        if next_ep:
            log.info("Tiếp tục chuỗi '%s' -> Tập %d/%d: %s", active["name"], next_ep["episode_num"], active["total_episodes"], next_ep["topic"])
            return {
                "series": active,
                "episode": next_ep,
                "series_context": {
                    "series_name": active["name"],
                    "episode_num": next_ep["episode_num"],
                    "total_episodes": active["total_episodes"],
                    "hook_from_previous": next_ep.get("hook_from_previous") or "",
                    "hook_to_next": next_ep.get("hook_to_next") or "",
                },
            }
        else:
            # Series đã hoàn thành tất cả các tập
            db.update_series(active["id"], status="completed")
            log.info("Series '%s' đã hoàn tất tất cả %d tập!", active["name"], active["total_episodes"])

    # DB trống hoặc series vừa xong -> Tự động tạo series mới!
    log.info("Database trống hoặc chuỗi trước đã hoàn tất -> Tự động khởi tạo Chuỗi Video Mới...")
    s_cfg = CONFIG.get("series", {})
    num_eps = int(s_cfg.get("default_episodes", 5))

    new_series = plan_new_series(num_episodes=num_eps)
    next_ep = db.get_next_pending_episode(new_series["id"])

    if not next_ep:
        raise RuntimeError("Khởi tạo series thất bại, không có tập nào được tạo.")

    return {
        "series": new_series,
        "episode": next_ep,
        "series_context": {
            "series_name": new_series["name"],
            "episode_num": next_ep["episode_num"],
            "total_episodes": new_series["total_episodes"],
            "hook_from_previous": next_ep.get("hook_from_previous") or "",
            "hook_to_next": next_ep.get("hook_to_next") or "",
        },
    }


def print_series_table() -> None:
    """Hiển thị bảng tiến độ các series trực quan trên terminal."""
    db.init_db()
    all_series = db.list_all_series()

    if not all_series:
        print("\n[Series Engine] Hiện chưa có chuỗi video nào trong database.\n")
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

    for s in all_series:
        st_text = status_icon.get(s["status"], s["status"])
        print(f"\n★ SERIES: {s['name']} ({st_text})")
        print(f"  Chủ đề: {s.get('theme', '')} | Chuyên ngành: {s.get('domain', '')} | Tiến độ: {s['completed_episodes']}/{s['total_episodes']} tập")
        print("  " + "-" * 74)

        for ep in s.get("episodes", []):
            ep_st = ep_icon.get(ep.get("status"), ep.get("status"))
            print(f"    Tập {ep['episode_num']:02d}: {ep['topic']:<45} {ep_st}")

    print("\n" + "=" * 78 + "\n")
