from __future__ import annotations
import json, logging, time
from pathlib import Path
from typing import Any
from urllib import request as _req

log = logging.getLogger(__name__)

_CACHE_FILE = Path(__file__).resolve().parent.parent / "output" / ".trend_cache.json"


def _cfg() -> dict:
    try:
        from .config import CONFIG
        return dict(CONFIG.get("trend", {}) or {})
    except Exception:
        return {}


def _cache_ttl() -> float:
    return float(_cfg().get("cache_hours", 6)) * 3600


def _sources() -> dict:
    return dict(_cfg().get("sources", {}) or {})


def _max_signals() -> int:
    return int(_cfg().get("max_signals", 18))


def _load_cache() -> dict[str, Any]:
    if not _CACHE_FILE.exists():
        return {}
    try:
        data = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        if time.time() - data.get("_ts", 0) < _cache_ttl():
            return data
    except Exception:
        pass
    return {}


def _save_cache(data: dict[str, Any]) -> None:
    _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    data["_ts"] = time.time()
    try:
        _CACHE_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _fetch_json(url: str, timeout: int = 8) -> Any:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; AutoVideoBot/1.0)"}
    r = _req.Request(url, headers=headers)
    with _req.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _hn_top(limit: int = 15) -> list[str]:
    try:
        ids: list[int] = _fetch_json(
            "https://hacker-news.firebaseio.com/v0/topstories.json", timeout=6
        )[:40]
        titles: list[str] = []
        for sid in ids:
            try:
                item = _fetch_json(
                    f"https://hacker-news.firebaseio.com/v0/item/{sid}.json", timeout=4
                )
                if item and item.get("type") == "story" and item.get("title"):
                    titles.append(item["title"])
                    if len(titles) >= limit:
                        break
            except Exception:
                continue
        log.debug("HN: %d tieu de", len(titles))
        return titles
    except Exception as e:
        log.warning("Hacker News loi: %s", e)
        return []


_TECH_SUBS = ["technology", "programming", "MachineLearning", "artificial", "compsci", "netsec"]


def _reddit_hot(subreddit: str, limit: int = 8) -> list[str]:
    url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}"
    try:
        data = _fetch_json(url, timeout=6)
        posts = data.get("data", {}).get("children", [])
        return [p["data"]["title"] for p in posts if p.get("data", {}).get("title")]
    except Exception as e:
        log.debug("Reddit r/%s loi: %s", subreddit, e)
        return []


def _reddit_top(limit_per_sub: int = 6) -> list[str]:
    titles: list[str] = []
    for sub in _TECH_SUBS:
        titles.extend(_reddit_hot(sub, limit_per_sub))
    return titles


def _google_trends(geo: str = "VN", n: int = 10) -> list[str]:
    try:
        from pytrends.request import TrendReq  # type: ignore
        pt = TrendReq(hl="vi-VN", tz=420, timeout=(6, 12), retries=1, backoff_factor=0.5)
        df = pt.trending_searches(pn="vietnam" if geo == "VN" else geo.lower())
        keywords = df[0].tolist()[:n]
        log.debug("Google Trends (%s): %s", geo, keywords)
        return [str(k) for k in keywords]
    except ImportError:
        log.debug("pytrends chua cai, bo qua Google Trends")
        return []
    except Exception as e:
        log.warning("Google Trends loi: %s", e)
        return []


def fetch_trends(force_refresh: bool = False) -> dict[str, list[str]]:
    """Tong hop trending signals tu tat ca nguon. Cache theo config cache_hours."""
    cfg = _cfg()
    if not cfg.get("enabled", True):
        log.debug("Trend disabled trong config")
        return {}

    if not force_refresh:
        cached = _load_cache()
        if cached:
            log.info("Dung trend cache (con han)")
            return {k: v for k, v in cached.items() if k != "_ts"}

    log.info("Lay trend moi tu HackerNews / Reddit / Google Trends...")
    src = _sources()
    result: dict[str, list[str]] = {}
    if src.get("hacker_news", True):
        result["hacker_news"] = _hn_top(15)
    if src.get("reddit", True):
        result["reddit"] = _reddit_top(6)
    if src.get("google_trends", True):
        result["google_trends"] = _google_trends("VN", 10)

    total = sum(len(v) for v in result.values())
    log.info("Trends: %d tin hieu (%s)", total, ", ".join(f"{k}={len(v)}" for k, v in result.items()))
    _save_cache(result)
    return result


def format_for_prompt(trends: dict[str, list[str]], max_items: int | None = None) -> str:
    """Dinh dang trends thanh doan van ngan de nhet vao prompt LLM."""
    if max_items is None:
        max_items = _max_signals()
    lines: list[str] = []

    hn = trends.get("hacker_news", [])[:max_items // 3]
    if hn:
        lines.append("Hacker News dang nong:")
        lines.extend(f"  - {t}" for t in hn)

    rd = trends.get("reddit", [])[:max_items // 3]
    if rd:
        lines.append("Reddit tech dang hot:")
        lines.extend(f"  - {t}" for t in rd)

    gt = trends.get("google_trends", [])[:max_items // 3]
    if gt:
        lines.append("Google Trends Viet Nam:")
        lines.extend(f"  - {t}" for t in gt)

    return "\n".join(lines) if lines else ""
