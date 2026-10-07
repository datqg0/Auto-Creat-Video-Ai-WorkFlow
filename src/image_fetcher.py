"""Tự tìm & tải ảnh minh họa cho scene dựa trên từ khóa.

Nguồn ảnh miễn phí, ưu tiên không cần API key:
  1. Openverse (CC, không cần key) - mặc định
  2. Pexels (cần PEXELS_API_KEY) - nếu có, chất lượng cao hơn

Ảnh tải về được cache theo từ khóa để lần sau khỏi tải lại. Nếu mọi nguồn
thất bại thì trả về None -> visual_engine vẽ nền gradient như cũ.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import re

import requests

from .config import CONFIG, env

log = logging.getLogger(__name__)

_CACHE = Path(__file__).resolve().parent.parent / "assets" / "image_cache"
_VIDEO_CACHE = Path(__file__).resolve().parent.parent / "assets" / "video_cache"
_TIMEOUT = 15
_VIDEO_TIMEOUT = 60
_HEADERS = {"User-Agent": "tech-video-bot/1.0"}


def _cache_path(query: str, index: int = 0) -> Path:
    key = hashlib.md5(f"{query.lower()}|{index}".encode("utf-8")).hexdigest()[:16]
    return _CACHE / f"{key}.jpg"


def _download(url: str, out: Path, headers: dict | None = None) -> bool:
    try:
        req_headers = dict(_HEADERS)
        if headers:
            req_headers.update(headers)
        r = requests.get(url, timeout=_TIMEOUT, headers=req_headers, stream=True)
        r.raise_for_status()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        # kiểm tra file ảnh hợp lệ
        if out.stat().st_size < 2000:
            out.unlink(missing_ok=True)
            return False
        return True
    except Exception as e:  # noqa: BLE001
        log.debug("Tải ảnh lỗi %s: %s", url, e)
        out.unlink(missing_ok=True)
        return False


def _search_google(query: str, count: int = 8) -> list[str]:
    """Tìm ảnh qua Google Custom Search API (nếu có GOOGLE_API_KEY + GOOGLE_CSE_ID)."""
    api_key = env("GOOGLE_API_KEY") or env("GEMINI_API_KEY")
    cse_id = env("GOOGLE_CSE_ID") or env("GOOGLE_SEARCH_CX")
    if not api_key or not cse_id:
        return []
    try:
        r = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                "key": api_key,
                "cx": cse_id,
                "q": query,
                "searchType": "image",
                "imgSize": "large",
                "imgType": "photo",
                "num": min(count, 10),
            },
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        items = r.json().get("items", [])
        urls = [it["link"] for it in items if it.get("link")]
        if urls:
            log.info("Google Image Search tìm thấy %d ảnh cho '%s'", len(urls), query)
        return urls
    except Exception as e:  # noqa: BLE001
        log.warning("Google Custom Search lỗi (%s) -> fallback", e)
        return []


def _search_duckduckgo(query: str, count: int = 8) -> list[str]:
    """Tìm ảnh qua DuckDuckGo / Bing web image search (100% miễn phí, không cần key)."""
    # 1. Thử qua thư viện ddgs / duckduckgo_search
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        with DDGS(timeout=10) as ddgs:
            results = list(ddgs.images(query, max_results=count))
            urls = [r["image"] for r in results if r.get("image")]
            if urls:
                log.info("DuckDuckGo Image Search tìm thấy %d ảnh cho '%s'", len(urls), query)
                return urls
    except Exception as e:  # noqa: BLE001
        log.debug("DuckDuckGo thư viện (%s) -> chuyển hướng web direct", e)

    # 2. Web direct search (Bing/DuckDuckGo index) - lọc ảnh ngang 16:9, không cần key
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.8",
        }
        r = requests.get(
            "https://www.bing.com/images/search",
            params={"q": query, "qft": "+filterui:aspect-wide"},
            headers=headers,
            timeout=_TIMEOUT,
        )
        if r.status_code == 200:
            matches = re.findall(r'm="({.*?})"', r.text)
            urls = []
            for m in matches[:count * 2]:
                try:
                    data = json.loads(m.replace("&quot;", '"'))
                    if "murl" in data and str(data["murl"]).startswith("http"):
                        urls.append(data["murl"])
                except Exception:
                    pass
            if urls:
                log.info("DuckDuckGo/Web Image Search tìm thấy %d ảnh cho '%s'", len(urls), query)
                return urls
    except Exception as e:  # noqa: BLE001
        log.debug("DuckDuckGo/Web direct search lỗi: %s", e)

    return []


def _search_pexels(query: str, count: int = 8) -> list[str]:
    key = env("PEXELS_API_KEY")
    if not key:
        return []
    try:
        r = requests.get(
            "https://api.pexels.com/v1/search",
            params={"query": query, "per_page": count, "orientation": "landscape"},
            headers={"Authorization": key},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        photos = r.json().get("photos", [])
        return [p["src"]["large2x"] for p in photos if p.get("src")]
    except Exception as e:  # noqa: BLE001
        log.debug("Pexels lỗi: %s", e)
    return []


def _search_openverse(query: str, count: int = 8) -> list[str]:
    try:
        r = requests.get(
            "https://api.openverse.org/v1/images/",
            params={
                "q": query,
                "page_size": count,
                "license_type": "all",
                "aspect_ratio": "wide",
                "mature": "false",
            },
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        return [it["url"] for it in results if it.get("url")]
    except Exception as e:  # noqa: BLE001
        log.debug("Openverse lỗi: %s", e)
    return []


def _search_tinyfish(query: str, count: int = 5) -> list[str]:
    """Tìm ảnh qua TinyFish AI Web Agent (chỉ chạy khi khả dụng và các nguồn khác đều xịt)."""
    try:
        from .tinyfish_client import is_available, fetch_images_with_tinyfish

        if not is_available():
            return []
        return fetch_images_with_tinyfish(query, count=count)
    except Exception as e:  # noqa: BLE001
        log.debug("TinyFish image search lỗi: %s", e)
        return []


def _format_ai_image_prompt(query: str, style: str = "kurzgesagt") -> str:
    """Tạo prompt chuẩn phong cách tech illustration/Kurzgesagt, triệt tiêu chữ rác."""
    cleaned = re.sub(r"[^\w\s\-\.\+]", " ", query).strip()
    if style == "kurzgesagt":
        return (
            f"Flat vector tech editorial illustration of {cleaned}. "
            "Kurzgesagt art style, bold saturated clean colors, smooth gradients, clean flat 2D shapes, "
            "soft neon cyber glow, conceptual technology art, dark vignette background. "
            "STRICT REQUIREMENT: Pure artwork without any text, no typography, no letters, no words, no subtitles, no watermark, no logo."
        )
    elif style == "blueprint":
        return (
            f"Technical schematic blueprint diagram of {cleaned}. "
            "Glowing cyan and neon circuit lines, dark blue technical grid, futuristic HUD interface. "
            "STRICT REQUIREMENT: No text, no letters, no words, no watermark."
        )
    else:  # realistic / 3d
        return (
            f"Cinematic 3D concept render of {cleaned}. "
            "Octane render, futuristic cyber technology aesthetic, dramatic volumetric lighting, 8k resolution. "
            "STRICT REQUIREMENT: No text, no letters, no words, no watermark."
        )


def generate_ai_image(
    prompt: str,
    out: Path | None = None,
    width: int | None = None,
    height: int | None = None,
    provider: str | None = None,
) -> Path | None:
    """Sinh 1 ảnh minh họa AI (1080p) theo prompt.

    Hỗ trợ các provider:
      1. Pollinations AI (FLUX) - 100% MIỄN PHÍ, KHÔNG CẦN KEY
      2. Hugging Face Inference (FLUX.1-schnell) nếu có HF_TOKEN
      3. Google Gemini (Imagen) nếu có GEMINI_API_KEY
    """
    w = width or int(CONFIG.get("visual", {}).get("width", 1920))
    h = height or int(CONFIG.get("visual", {}).get("height", 1080))
    img_cfg = CONFIG.get("images", {})
    style = img_cfg.get("ai_style", "kurzgesagt")
    prov = (provider or img_cfg.get("ai_provider", "pollinations")).lower()

    full_prompt = _format_ai_image_prompt(prompt, style=style)
    target_out = out or _cache_path(prompt, 0)
    target_out.parent.mkdir(parents=True, exist_ok=True)

    # 1. Thử HuggingFace nếu được chỉ định và có token
    if prov == "huggingface":
        token = env("HF_TOKEN")
        if token:
            try:
                from huggingface_hub import InferenceClient

                client = InferenceClient(api_key=token)
                img = client.text_to_image(full_prompt, model="black-forest-labs/FLUX.1-schnell", width=w, height=h)
                img.save(target_out, format="JPEG", quality=90)
                if target_out.exists() and target_out.stat().st_size > 2000:
                    return target_out
            except Exception as e:
                log.warning("Hugging Face sinh ảnh lỗi (%s) -> fallback Pollinations", e)

    # 2. Thử Gemini nếu được chỉ định và có key
    if prov == "gemini":
        gemini_key = env("GEMINI_API_KEY")
        if gemini_key:
            try:
                from google import genai

                client = genai.Client(api_key=gemini_key)
                resp = client.models.generate_content(
                    model="gemini-2.5-flash-image",
                    contents=full_prompt,
                )
                for part in resp.candidates[0].content.parts:
                    inline = getattr(part, "inline_data", None)
                    if inline and inline.data:
                        import io
                        from PIL import Image

                        img = Image.open(io.BytesIO(inline.data)).convert("RGB")
                        img.save(target_out, format="JPEG", quality=90)
                        if target_out.exists() and target_out.stat().st_size > 2000:
                            return target_out
            except Exception as e:
                log.warning("Gemini sinh ảnh lỗi (%s) -> fallback Pollinations", e)

    # 3. Mặc định / Fallback số 1: Pollinations AI (FLUX) - Hoàn toàn miễn phí, không cần key
    try:
        import urllib.parse

        negative_clause = urllib.parse.quote(
            "text,letters,words,typography,watermark,logo,captions,labels,noisy,blurry,deformed,gibberish,symbols"
        )
        encoded_prompt = urllib.parse.quote(full_prompt)
        seed = abs(hash(f"{prompt}_{w}_{h}")) % 1000000
        # Pollinations free tier cho phép tối đa 1280x720 (ngang) hoặc 720x1280 (dọc)
        is_vert = h > w
        poll_w = 720 if is_vert else 1280
        poll_h = 1280 if is_vert else 720
        url = (
            f"https://image.pollinations.ai/prompt/{encoded_prompt}"
            f"?width={poll_w}&height={poll_h}&model=flux&nologo=true&seed={seed}&negative={negative_clause}"
        )
        auth_headers = {}
        poll_key = env("POLLINATIONS_API_KEY")
        if poll_key and "your_" not in poll_key:
            auth_headers["Authorization"] = f"Bearer {poll_key}"
        if _download(url, target_out, headers=auth_headers):
            # Co dãn sắc nét sang kích thước chuẩn (1920x1080) bằng Lanczos
            if (w, h) != (poll_w, poll_h):
                try:
                    from PIL import Image

                    with Image.open(target_out) as im:
                        im_resized = im.resize((w, h), Image.Resampling.LANCZOS)
                        im_resized.save(target_out, format="JPEG", quality=92)
                except Exception as e:
                    log.debug("Resize ảnh AI lỗi: %s", e)
            return target_out
    except Exception as e:
        log.warning("Pollinations sinh ảnh lỗi: %s", e)

    return None


def fetch_image(query: str, index: int = 0) -> Path | None:
    """Trả về ảnh thứ ``index`` cho từ khóa (cho phép nhiều ảnh khác nhau/1 từ khóa).

    Chiến lược:
      - Nếu bật ``images.ai_primary: true`` -> Ưu tiên sinh ảnh AI Kurzgesagt/FLUX trước.
      - Sau đó thử tìm ảnh stock: Google -> DuckDuckGo -> Pexels -> Openverse -> TinyFish.
      - Nếu không tìm được stock và bật ``images.ai_enabled`` (mặc định) -> Tự động sinh ảnh AI bám sát chủ đề!
    """
    query = (query or "").strip()
    if not query:
        return None

    cached = _cache_path(query, index)
    if cached.exists():
        return cached

    img_cfg = CONFIG.get("images", {})
    ai_enabled = img_cfg.get("ai_enabled", True)
    ai_primary = img_cfg.get("ai_primary", False)
    w = int(CONFIG.get("visual", {}).get("width", 1920))
    h = int(CONFIG.get("visual", {}).get("height", 1080))

    # 1. Nếu ưu tiên AI -> sinh ảnh AI trước
    if ai_enabled and ai_primary:
        ai_res = generate_ai_image(query, out=cached, width=w, height=h)
        if ai_res and ai_res.exists():
            log.info("Ảnh AI (Primary) '%s' #%d -> %s", query, index, cached.name)
            return cached

    # 2. Tìm kiếm ảnh stock từ các nguồn
    for search in (_search_google, _search_duckduckgo, _search_pexels, _search_openverse, _search_tinyfish):
        urls = search(query)
        if not urls:
            continue
        url = urls[index] if index < len(urls) else urls[index % len(urls)]
        if _download(url, cached):
            log.info("Ảnh stock '%s' #%d -> %s", query, index, cached.name)
            return cached

    # 3. Tự động fallback: Sinh ảnh AI khi stock photo không có
    if ai_enabled:
        log.info("Stock không có ảnh cho '%s' #%d -> Kích hoạt AI sinh ảnh minh họa...", query, index)
        ai_res = generate_ai_image(query, out=cached, width=w, height=h)
        if ai_res and ai_res.exists():
            log.info("Ảnh AI (Fallback) '%s' #%d -> %s", query, index, cached.name)
            return cached

    log.info("Không tìm được ảnh cho '%s' #%d, dùng nền gradient", query, index)
    return None


# ------------------------------ VIDEO b-roll ------------------------------
# Tải clip footage động minh họa (Pexels Videos). Cần PEXELS_API_KEY.
# Nếu không có key hoặc tải lỗi -> trả None, pipeline dùng ảnh tĩnh như cũ.

def _video_cache_path(query: str, index: int, orientation: str) -> Path:
    key = hashlib.md5(f"vid|{query.lower()}|{index}|{orientation}".encode("utf-8")).hexdigest()[:16]
    return _VIDEO_CACHE / f"{key}.mp4"


def _download_video(url: str, out: Path) -> bool:
    try:
        r = requests.get(url, timeout=_VIDEO_TIMEOUT, headers=_HEADERS, stream=True)
        r.raise_for_status()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                f.write(chunk)
        if out.stat().st_size < 20000:  # file quá nhỏ = hỏng
            out.unlink(missing_ok=True)
            return False
        return True
    except Exception as e:  # noqa: BLE001
        log.debug("Tải video lỗi %s: %s", url, e)
        out.unlink(missing_ok=True)
        return False


def _search_pexels_videos(query: str, orientation: str, count: int = 10) -> list[str]:
    """Trả về danh sách link file .mp4 từ Pexels Videos (ưu tiên độ phân giải HD gần 1080)."""
    key = env("PEXELS_API_KEY")
    if not key:
        return []
    try:
        r = requests.get(
            "https://api.pexels.com/videos/search",
            params={"query": query, "per_page": count, "orientation": orientation},
            headers={"Authorization": key},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        out: list[str] = []
        for v in r.json().get("videos", []):
            files = v.get("video_files", [])
            if not files:
                continue
            # chọn file .mp4 có chiều cao gần 1080 nhất (không quá lớn để tải nhanh)
            mp4s = [f for f in files if f.get("file_type") == "video/mp4" and f.get("link")]
            if not mp4s:
                continue
            best = min(mp4s, key=lambda f: abs((f.get("height") or 0) - 1080))
            out.append(best["link"])
        return out
    except Exception as e:  # noqa: BLE001
        log.debug("Pexels Videos lỗi: %s", e)
    return []


def fetch_video(query: str, index: int = 0, vertical: bool = False) -> Path | None:
    """Tải clip b-roll cho từ khóa. ``vertical`` True cho short 9:16.

    Trả về đường dẫn .mp4 đã cache, hoặc None nếu không có nguồn/không tải được.
    """
    query = (query or "").strip()
    if not query:
        return None
    orientation = "portrait" if vertical else "landscape"
    cached = _video_cache_path(query, index, orientation)
    if cached.exists():
        return cached

    urls = _search_pexels_videos(query, orientation)
    if not urls:
        return None
    url = urls[index] if index < len(urls) else urls[index % len(urls)]
    if _download_video(url, cached):
        log.info("Video b-roll '%s' #%d (%s) -> %s", query, index, orientation, cached.name)
        return cached
    return None
