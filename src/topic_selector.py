"""AI tu chon chu de tech, tranh trung voi cac chu de da dung gan day.
Tich hop du lieu trending (HackerNews, Reddit, Google Trends) lam ngu canh.
"""
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

    prompt = f"""Ban la bien tap vien kenh YouTube ve cong nghe theo phong cach giai thich truc quan (visualization).
Hay de xuat MOT chu de video cu the, hap dan trong linh vuc: {domain}.

Yeu cau:
- Chu de du hep de giai thich trong video 5 phut bang hinh anh dong/bieu do.
- Phu hop de minh hoa bang animation, bieu do du lieu, hoac mo phong thuat toan.
- Viet bang {lang_name}.
- KHONG trung hoac qua giong cac chu de da dung duoi day:
{used_block}{trend_section}

Chi tra ve DUY NHAT ten chu de tren mot dong, khong giai thich, khong danh so, khong dau ngoac kep."""

    topic = generate(prompt).strip().splitlines()[0].strip().strip('"').strip("-").strip()
    log.info("Chu de da chon: %s", topic)
    return topic
