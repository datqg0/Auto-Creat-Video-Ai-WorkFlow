"""TinyFish AI Agent Client (Web Automation & Intelligent Scraping).

Chỉ kích hoạt khi TINYFISH_API_KEY khả dụng trong môi trường hoặc cấu hình.
Cung cấp 4 ứng dụng chính:
  1. research_topic_context: Thu thập dữ liệu thực tế, định nghĩa & testcase chuẩn để làm giàu prompt kịch bản.
  2. fetch_images_with_tinyfish: Fallback tìm ảnh chất lượng cao khi Google / DuckDuckGo bị 403.
  3. fetch_github_trending: Cào kho lưu trữ thịnh hành từ GitHub Trending cho trend_fetcher.
  4. research_youtube_competitors: Nghiên cứu tiêu đề và thumbnail của đối thủ trên YouTube.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.parse
from typing import Any

from dotenv import load_dotenv
import requests

from .config import env

load_dotenv()

log = logging.getLogger(__name__)

_ENDPOINT = "https://agent.tinyfish.ai/v1/automation/run-sse"
_DEFAULT_TIMEOUT = 90
_API_KEY_ENV = "TINYFISH_API_KEY"

# Lưu trạng thái tạm thời nếu API key bị lỗi xác thực để tránh gọi lại vô ích
_TINYFISH_DISABLED = False


def is_available() -> bool:
    """Kiểm tra xem TinyFish có sẵn sàng và có API key hợp lệ không."""
    global _TINYFISH_DISABLED
    if _TINYFISH_DISABLED:
        return False
    key = env(_API_KEY_ENV) or os.getenv(_API_KEY_ENV)
    return bool(key and key.strip().startswith("sk-tinyfish-"))


def run_automation(url: str, goal: str, timeout: int = _DEFAULT_TIMEOUT) -> Any | None:
    """Gửi yêu cầu điều khiển web agent tới TinyFish và lắng nghe SSE kết quả.
    
    Chỉ chạy khi is_available() == True. Trả về kết quả JSON hoặc None nếu lỗi.
    """
    global _TINYFISH_DISABLED
    if not is_available():
        return None

    api_key = os.getenv(_API_KEY_ENV, "").strip()
    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "url": url,
        "goal": goal,
    }

    try:
        log.info("TinyFish Agent bắt đầu tác vụ tại %s: '%s'", url[:60], goal[:80])
        resp = requests.post(
            _ENDPOINT,
            headers=headers,
            json=payload,
            stream=True,
            timeout=timeout,
        )

        if resp.status_code in (401, 403):
            log.warning("TinyFish API key không hợp lệ hoặc hết hạn (status %d) -> tạm vô hiệu hóa", resp.status_code)
            _TINYFISH_DISABLED = True
            return None

        resp.raise_for_status()

        final_result = None
        for line in resp.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8", errors="replace")
            if decoded.startswith("data: "):
                try:
                    event = json.loads(decoded[6:])
                    event_type = event.get("type")
                    if event_type == "COMPLETE":
                        status = event.get("status")
                        if status == "COMPLETED":
                            raw_res = event.get("result")
                            # Có thể result bọc trong {"result": ...}
                            if isinstance(raw_res, dict) and "result" in raw_res:
                                final_result = raw_res["result"]
                            else:
                                final_result = raw_res
                        break
                    elif event_type == "ERROR":
                        log.warning("TinyFish báo lỗi tác vụ: %s", event.get("message"))
                        break
                except Exception:
                    continue

        return final_result

    except Exception as e:
        log.debug("TinyFish automation thất bại hoặc quá thời gian (%s)", e)
        return None


# ---------------------------------------------------------------------------
# ỨNG DỤNG 1: Thu thập tư liệu thực tế & định nghĩa kỹ thuật cho Kịch bản
# ---------------------------------------------------------------------------
def research_topic_context(topic: str) -> str:
    """Dùng TinyFish cào tóm tắt kiến thức / thuật toán thực tế từ Wikipedia/Web để bổ trợ viết kịch bản."""
    if not is_available() or not topic or not topic.strip():
        return ""

    encoded = urllib.parse.quote(topic.strip())
    # Sử dụng Wikipedia tiếng Anh hoặc tìm kiếm để có thông tin kỹ thuật chuẩn xác nhất
    search_url = f"https://en.wikipedia.org/wiki/{encoded}"
    goal = (
        f"Extract key factual details about '{topic}': "
        "1. Core concept definition. 2. Time and space complexity (if applicable). "
        "3. Real-world applications or famous examples. "
        "Return a clean JSON object with keys: concept, complexity, applications."
    )

    data = run_automation(search_url, goal, timeout=75)
    if not data:
        # Fallback thử qua DuckDuckGo HTML search nếu Wikipedia không khớp URL
        ddg_url = f"https://html.duckduckgo.com/html/?q={encoded}+computer+science+algorithm+overview"
        goal_fallback = "Extract the top 2 search result snippets explaining the concept and Big-O complexity. Return JSON with 'summary'."
        data = run_automation(ddg_url, goal_fallback, timeout=60)

    if not data:
        return ""

    try:
        if isinstance(data, dict):
            parts = []
            if data.get("concept"):
                parts.append(f"- Khái niệm: {data['concept']}")
            if data.get("complexity"):
                parts.append(f"- Độ phức tạp: {data['complexity']}")
            if data.get("applications"):
                apps = data["applications"]
                parts.append(f"- Ứng dụng thực tế: {apps if isinstance(apps, str) else ', '.join(map(str, apps))}")
            if data.get("summary"):
                parts.append(f"- Tóm lược: {data['summary']}")
            if parts:
                return "\n".join(parts)
        elif isinstance(data, str):
            return data[:500]
    except Exception as e:
        log.debug("Lỗi xử lý research_topic_context: %s", e)

    return ""


# ---------------------------------------------------------------------------
# ỨNG DỤNG 2: Fallback tìm ảnh khi Google & DuckDuckGo bị lỗi / 403
# ---------------------------------------------------------------------------
def fetch_images_with_tinyfish(query: str, count: int = 5) -> list[str]:
    """Tìm ảnh chất lượng cao trên Unsplash qua TinyFish khi Google/DuckDuckGo bị 403."""
    if not is_available() or not query or not query.strip():
        return []

    encoded = urllib.parse.quote(query.strip())
    url = f"https://unsplash.com/s/photos/{encoded}"
    goal = (
        f"Extract direct URLs of the first {min(count, 5)} landscape photos matching '{query}'. "
        "Look for img tags with https://images.unsplash.com. Return JSON array of image URL strings."
    )

    data = run_automation(url, goal, timeout=60)
    if not data:
        return []

    urls: list[str] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str) and item.startswith("http"):
                urls.append(item)
            elif isinstance(item, dict) and "url" in item and str(item["url"]).startswith("http"):
                urls.append(item["url"])
    elif isinstance(data, dict):
        for val in data.values():
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, str) and item.startswith("http"):
                        urls.append(item)

    if urls:
        log.info("TinyFish Image Search tìm thấy %d ảnh cho '%s'", len(urls), query)
    return urls[:count]


# ---------------------------------------------------------------------------
# ỨNG DỤNG 3: Thu thập Trend từ GitHub Trending cho trend_fetcher
# ---------------------------------------------------------------------------
def fetch_github_trending(limit: int = 6) -> list[str]:
    """Cào danh sách repository đang thịnh hành trên GitHub Trending."""
    if not is_available():
        return []

    url = "https://github.com/trending"
    goal = (
        f"Extract the top {limit} trending repositories. "
        "For each repository, get the repository name (owner/repo) and a short description. "
        "Return a JSON list of objects with 'name' and 'description'."
    )

    data = run_automation(url, goal, timeout=60)
    if not data or not isinstance(data, list):
        return []

    trends: list[str] = []
    for item in data[:limit]:
        if isinstance(item, dict):
            name = item.get("name") or item.get("repo") or item.get("title")
            desc = item.get("description") or item.get("desc") or ""
            if name:
                trends.append(f"{name}: {desc}".strip(" :"))
        elif isinstance(item, str):
            trends.append(item)

    if trends:
        log.info("TinyFish cào thành công %d GitHub Trending repos", len(trends))
    return trends


# ---------------------------------------------------------------------------
# ỨNG DỤNG 4: Nghiên cứu đối thủ YouTube để tối ưu SEO & Thumbnail
# ---------------------------------------------------------------------------
def research_youtube_competitors(topic: str, limit: int = 3) -> dict[str, Any] | None:
    """Nghiên cứu top video YouTube cùng chủ đề để lấy tiêu đề nhiều view và phong cách thumbnail."""
    if not is_available() or not topic or not topic.strip():
        return None

    encoded = urllib.parse.quote(topic.strip())
    url = f"https://www.youtube.com/results?search_query={encoded}"
    goal = (
        f"Extract the top {limit} video results for '{topic}'. "
        "For each video, extract: 1. Video title, 2. View count, 3. Thumbnail image URL. "
        "Return JSON list of objects with 'title', 'views', and 'thumbnail'."
    )

    data = run_automation(url, goal, timeout=70)
    if not data or not isinstance(data, list):
        return None

    competitors = []
    for item in data[:limit]:
        if isinstance(item, dict) and item.get("title"):
            competitors.append({
                "title": item.get("title"),
                "views": item.get("views", "N/A"),
                "thumbnail": item.get("thumbnail"),
            })

    if competitors:
        log.info("TinyFish phân tích được %d video đối thủ trên YouTube cho '%s'", len(competitors), topic)
        return {"competitors": competitors}

    return None
