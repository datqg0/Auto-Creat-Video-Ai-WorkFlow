"""Hệ thống nghiên cứu tài liệu kỹ thuật & kiến thức cho kịch bản và hoạt hoạ.

Thực hiện cơ chế 3 Tầng Fallback hoàn chỉnh:
  - Tầng 1 (Tự động tìm & tải trên mạng):
      + Wikipedia REST / Action API (tiếng Việt & tiếng Anh)
      + DuckDuckGo Instant Answer & Web Snippets
      + ArXiv API (cho các chủ đề AI / Computer Science chuyên sâu)
      + TinyFish Web Agent (nếu có API key)
  - Tầng 2 (Tải từ danh sách Link Backup uy tín):
      + Đọc cấu hình từ assets/docs/sources.yaml hoặc config.yaml
      + Khớp từ khóa chủ đề và tải trực tiếp từ URL chuẩn
  - Tầng 3 (Kho tài liệu chuẩn bị sẵn offline - Tải sẵn):
      + Quét và trích xuất dữ liệu chuẩn 100% (Zero Hallucination) từ assets/docs/*.md
      + Hỗ trợ trích xuất file PDF nếu có trong thư mục assets/docs/
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import urllib.parse
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml

from .config import CONFIG

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
_DOCS_DIR = _ROOT / "assets" / "docs"
_CACHE_DIR = _ROOT / "assets" / "cache" / "research"
_SOURCES_YAML = _DOCS_DIR / "sources.yaml"

_USER_AGENT = "AutoVideoAI-ResearchBot/2.0 (Education; High-Accuracy Visualizer; +https://github.com/datqg0/Auto-Creat-Video-Ai-WorkFlow)"
_DEFAULT_TIMEOUT = 10


@dataclass
class ResearchDocument:
    """Đối tượng lưu trữ tài liệu nghiên cứu đã thu thập và cấu trúc hóa."""
    topic: str
    tier: str  # tier1_auto_search | tier2_backup_link | tier3_offline_doc
    source_name: str
    source_url: str = ""
    title: str = ""
    concept: str = ""
    math_formulas: List[str] = field(default_factory=list)
    sample_data: str = ""
    steps: List[str] = field(default_factory=list)
    complexity: str = ""
    full_text: str = ""

    def to_prompt_block(self, max_chars: int = 4000) -> str:
        """Định dạng tài liệu thành khối chuẩn <research> để đưa vào LLM Prompt.
        
        Kèm các quy tắc bắt buộc LLM và Animation Runner phải 'làm theo tài liệu'.
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f"TÀI LIỆU KHOA HỌC & DỮ LIỆU ĐÃ XÁC THỰC (BẮT BUỘC 'LÀM THEO TÀI LIỆU')")
        lines.append(f"[Cơ chế: {self.tier.upper()} | Nguồn: {self.source_name}]")
        if self.source_url:
            lines.append(f"[Tham chiếu: {self.source_url}]")
        lines.append("=" * 80)
        lines.append("<research>")

        if self.title:
            lines.append(f"CHỦ ĐỀ CHUẨN: {self.title}")
        if self.concept:
            lines.append(f"\n1. KHÁI NIỆM & BẢN CHẤT:\n{self.concept.strip()}")
        if self.math_formulas:
            lines.append("\n2. CÔNG THỨC TOÁN HỌC & SỐ LIỆU CHÍNH XÁC (ZERO HALLUCINATION):")
            for f in self.math_formulas:
                lines.append(f"   • {f}")
        if self.sample_data:
            lines.append(f"\n3. BỘ DỮ LIỆU MẪU & MẢNG MINH HỌA (DÙNG CHO HOẠT HOẠ):\n{self.sample_data.strip()}")
        if self.steps:
            lines.append("\n4. CÁC BƯỚC / THUẬT TOÁN TỪNG BƯỚC:")
            for i, st in enumerate(self.steps, 1):
                lines.append(f"   {i}. {st}")
        if self.complexity:
            lines.append(f"\n5. ĐỘ PHỨC TẠP (BIG-O):\n{self.complexity.strip()}")

        if self.full_text and not (self.concept or self.sample_data):
            lines.append(f"\nNỘI DUNG TÀI LIỆU:\n{self.full_text.strip()}")

        lines.append("</research>")
        lines.append("=" * 80)
        lines.append("QUY TẮC BẮT BUỘC CHO KỊCH BẢN & ANIMATION ('LÀM THEO TÀI LIỆU'):")
        lines.append("1. LỜI THOẠI (NARRATION): Phải dùng đúng các con số, định nghĩa và công thức trong thẻ <research>.")
        lines.append("   Tuyệt đối không bịa đặt số tính nhầm lẫn (ví dụ với RSA, phép tính và số dư phải khớp 100%).")
        lines.append("2. HOẠT HOẠ (VISUAL PROMPT & ANIMATION SPEC): Phải trực quan hóa đúng mảng dữ liệu, ma trận")
        lines.append("   hoặc bước nhảy trạng thái đã nêu trong tài liệu trên. CẤM gán bừa đồ thị giải tích hay ma trận của bài khác.")
        lines.append("3. THỬ THÁCH (EXERCISES): Đặt câu hỏi và gợi ý dựa trên đúng các ràng buộc của tài liệu.")
        lines.append("=" * 80)

        block = "\n".join(lines)
        if len(block) > max_chars:
            block = block[:max_chars].rstrip() + "\n[...Đã trích xuất phần cốt lõi]\n</research>"
        return block


# ---------------------------------------------------------------------------
# CÁC HÀM BỔ TRỢ & CACHE
# ---------------------------------------------------------------------------

def _clean_keyword(text: str) -> str:
    """Lọc từ khóa kỹ thuật cốt lõi từ tiêu đề/chủ đề tiếng Việt."""
    t = text.strip()
    # Nếu có dấu hai chấm (VD: "Bí mật sạc nhanh: Tại sao..."), lấy vế trước hoặc vế trọng tâm
    if ":" in t:
        parts = t.split(":")
        first = parts[0].strip()
        second = parts[1].strip()
        # Ưu tiên vế ngắn gọn chứa chủ thể
        t = first if 4 <= len(first) <= 40 else second

    t = t.lower()
    # Loại bỏ các tiền tố / từ giật gân thường gặp
    clickbait = (
        r"\b(tại sao|vì sao|làm sao|cơ chế|giải mã|tìm hiểu|bí mật|nghệ thuật|hành trình|"
        r"thuật toán|cấu trúc|kiến trúc|kỳ diệu|rùa bò|phát nổ|ngon lành|chớp mắt|bất tử|"
        r"như thế nào|là gì|siêu tốc|toàn tập|hướng dẫn)\b"
    )
    t = re.sub(clickbait, " ", t)
    # Bỏ dấu chấm hỏi, ngoặc
    t = re.sub(r"[?!:,\"\']", " ", t)
    cleaned = " ".join(t.split())
    return cleaned if len(cleaned) >= 3 else text.strip()


def _get_cache_path(topic: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-zA-Z0-9_\-]+", "_", topic.strip().lower())[:60]
    md5 = hashlib.md5(topic.strip().encode("utf-8")).hexdigest()[:8]
    return _CACHE_DIR / f"{slug}_{md5}.json"


def _load_cached_research(topic: str) -> Optional[ResearchDocument]:
    cfg = CONFIG.get("research", {}) or {}
    if not cfg.get("cache_enabled", True):
        return None
    cache_file = _get_cache_path(topic)
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            log.info("Research Cache Hit cho chủ đề '%s' (Nguồn: %s, Tầng: %s)", topic, data.get("source_name"), data.get("tier"))
            return ResearchDocument(**data)
        except Exception as e:
            log.debug("Đọc cache research lỗi: %s", e)
    return None


def _save_cached_research(doc: ResearchDocument) -> None:
    try:
        cache_file = _get_cache_path(doc.topic)
        cache_file.write_text(json.dumps(asdict(doc), ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        log.debug("Lưu cache research lỗi: %s", e)


def _safe_request_get(url: str, params: dict | None = None, timeout: int = _DEFAULT_TIMEOUT) -> Optional[requests.Response]:
    """Gửi HTTP request an toàn, tự động xử lý certifi và header."""
    headers = {"User-Agent": _USER_AGENT, "Accept": "application/json, text/html, */*"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            return resp
    except Exception as e:
        # Thử lại 1 lần với verify=False nếu gặp lỗi SSL cục bộ máy khách
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout, verify=False)
            if resp.status_code == 200:
                return resp
        except Exception:
            pass
        log.debug("Safe request failed for %s: %s", url[:60], e)
    return None


# ---------------------------------------------------------------------------
# TẦNG 1: TỰ ĐỘNG TÌM & TẢI TRÊN MẠNG (Auto Search & Download)
# ---------------------------------------------------------------------------

def _search_wikipedia_api(query: str, lang: str = "en") -> Optional[Dict[str, Any]]:
    """Tìm kiếm trang Wikipedia và lấy bản tóm tắt nội dung + metadata."""
    clean_q = _clean_keyword(query)
    q_words = [w for w in re.findall(r"\w+", clean_q.lower()) if len(w) > 2]

    # 1. Tìm kiếm danh sách tiêu đề khớp nhất
    search_url = f"https://{lang}.wikipedia.org/w/api.php"
    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": clean_q,
        "format": "json",
        "utf8": 1,
        "srlimit": 4,
    }
    resp = _safe_request_get(search_url, params=search_params)
    if not resp:
        return None

    try:
        data = resp.json()
        search_results = data.get("query", {}).get("search", [])
        if not search_results:
            return None

        # Duyệt qua các kết quả, kiểm tra độ liên quan để loại bỏ các trang trật lỏ (như Doraemon / We Are the World)
        best_item = None
        for item in search_results:
            title_lower = item["title"].lower()
            snippet_lower = item.get("snippet", "").lower()
            # Nếu là Wikipedia tiếng Việt mà title hoàn toàn không có từ khóa nào trong query, bỏ qua
            if lang == "vi" and q_words and not any(w in title_lower for w in q_words):
                continue
            if any(w in title_lower or w in snippet_lower for w in q_words):
                best_item = item
                break

        if not best_item:
            return None

        best_title = best_item["title"]

        # 2. Lấy thông tin trích xuất tóm tắt qua REST API
        encoded_title = urllib.parse.quote(best_title.replace(" ", "_"))
        summary_url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{encoded_title}"
        sum_resp = _safe_request_get(summary_url)
        if sum_resp:
            sum_data = sum_resp.json()
            extract = sum_data.get("extract", "")
            if extract and len(extract) > 100:
                return {
                    "title": sum_data.get("title", best_title),
                    "extract": extract,
                    "url": sum_data.get("content_urls", {}).get("desktop", {}).get("page", f"https://{lang}.wikipedia.org/wiki/{encoded_title}"),
                    "description": sum_data.get("description", ""),
                    "source": f"Wikipedia ({lang.upper()})",
                }
    except Exception as e:
        log.debug("Wikipedia API parse lỗi: %s", e)
    return None


def _search_duckduckgo_api(query: str) -> Optional[Dict[str, Any]]:
    """Tra cứu DuckDuckGo Instant Answer API."""
    clean_q = _clean_keyword(query)
    url = "https://api.duckduckgo.com/"
    params = {
        "q": clean_q,
        "format": "json",
        "no_html": 1,
        "skip_disambig": 1,
    }
    resp = _safe_request_get(url, params=params)
    if not resp:
        return None

    try:
        data = resp.json()
        abstract = data.get("AbstractText", "")
        heading = data.get("Heading", clean_q)
        source_url = data.get("AbstractURL", "")
        if abstract and len(abstract) > 60:
            return {
                "title": heading,
                "extract": abstract,
                "url": source_url,
                "source": "DuckDuckGo Instant Answer",
            }

        # Nếu không có Abstract, kiểm tra RelatedTopics
        related = data.get("RelatedTopics", [])
        snippets = []
        for r in related[:3]:
            if isinstance(r, dict) and r.get("Text"):
                snippets.append(r["Text"])
        if snippets:
            return {
                "title": heading,
                "extract": "\n".join(snippets),
                "url": source_url or "https://duckduckgo.com",
                "source": "DuckDuckGo Related Topics",
            }
    except Exception as e:
        log.debug("DuckDuckGo parse lỗi: %s", e)
    return None


def _search_arxiv_api(query: str) -> Optional[Dict[str, Any]]:
    """Tra cứu kho bài báo khoa học ArXiv cho các chủ đề AI / ML / Math sâu."""
    clean_q = _clean_keyword(query)
    url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{clean_q}",
        "max_results": 1,
    }
    resp = _safe_request_get(url, params=params)
    if not resp:
        return None

    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "xml")
        entry = soup.find("entry")
        if entry:
            title = entry.find("title")
            summary = entry.find("summary")
            entry_id = entry.find("id")
            if summary and len(summary.text.strip()) > 100:
                return {
                    "title": title.text.strip() if title else clean_q,
                    "extract": summary.text.strip().replace("\n", " "),
                    "url": entry_id.text.strip() if entry_id else "https://arxiv.org",
                    "source": "ArXiv Research Paper",
                }
    except Exception as e:
        log.debug("ArXiv parse lỗi: %s", e)
    return None


def _fetch_tier1_auto_search(topic: str) -> Optional[ResearchDocument]:
    """Thực thi Tầng 1: Tự động tìm kiếm & tải tài liệu trên mạng."""
    log.info("▶ [Research Tầng 1] Đang tự động tìm tài liệu trên mạng cho: '%s'", topic)

    # 1. Thử Wikipedia tiếng Việt
    res_vi = _search_wikipedia_api(topic, lang="vi")
    if res_vi and len(res_vi["extract"]) > 180:
        log.info("✓ [Research Tầng 1] Tìm thấy trên Wikipedia tiếng Việt: '%s'", res_vi["title"])
        return ResearchDocument(
            topic=topic,
            tier="tier1_auto_search",
            source_name=res_vi["source"],
            source_url=res_vi.get("url", ""),
            title=res_vi["title"],
            concept=res_vi["extract"],
            full_text=res_vi["extract"],
        )

    # 2. Thử Wikipedia tiếng Anh (thường có dữ liệu thuật toán/kỹ thuật đầy đủ nhất)
    res_en = _search_wikipedia_api(topic, lang="en")
    if res_en and len(res_en["extract"]) > 180:
        log.info("✓ [Research Tầng 1] Tìm thấy trên Wikipedia tiếng Anh: '%s'", res_en["title"])
        return ResearchDocument(
            topic=topic,
            tier="tier1_auto_search",
            source_name=res_en["source"],
            source_url=res_en.get("url", ""),
            title=res_en["title"],
            concept=res_en["extract"],
            full_text=res_en["extract"],
        )

    # 3. Thử DuckDuckGo Instant Answer
    res_ddg = _search_duckduckgo_api(topic)
    if res_ddg and len(res_ddg["extract"]) > 120:
        log.info("✓ [Research Tầng 1] Tìm thấy trên DuckDuckGo: '%s'", res_ddg["title"])
        return ResearchDocument(
            topic=topic,
            tier="tier1_auto_search",
            source_name=res_ddg["source"],
            source_url=res_ddg.get("url", ""),
            title=res_ddg["title"],
            concept=res_ddg["extract"],
            full_text=res_ddg["extract"],
        )

    # 4. Thử ArXiv (chuyên khảo AI/CS)
    res_arxiv = _search_arxiv_api(topic)
    if res_arxiv and len(res_arxiv["extract"]) > 150:
        log.info("✓ [Research Tầng 1] Tìm thấy trên ArXiv: '%s'", res_arxiv["title"])
        return ResearchDocument(
            topic=topic,
            tier="tier1_auto_search",
            source_name=res_arxiv["source"],
            source_url=res_arxiv.get("url", ""),
            title=res_arxiv["title"],
            concept=res_arxiv["extract"],
            full_text=res_arxiv["extract"],
        )

    # 5. Thử TinyFish Web Agent nếu có key
    try:
        from .tinyfish_client import is_available, research_topic_context
        if is_available():
            tf_data = research_topic_context(topic)
            if tf_data and len(tf_data) > 100:
                log.info("✓ [Research Tầng 1] Tìm thấy qua TinyFish Agent cho '%s'", topic)
                return ResearchDocument(
                    topic=topic,
                    tier="tier1_auto_search",
                    source_name="TinyFish AI Web Agent",
                    title=topic,
                    concept=tf_data,
                    full_text=tf_data,
                )
    except Exception as e:
        log.debug("TinyFish fallback lỗi: %s", e)

    log.warning("✗ [Research Tầng 1] Không tìm thấy dữ liệu mạng thỏa mãn -> chuyển sang Tầng 2 (Backup List)")
    return None


# ---------------------------------------------------------------------------
# TẦNG 2: TẢI THEO DANH SÁCH LINK BACKUP (Curated Backup Links)
# ---------------------------------------------------------------------------

def _load_sources_yaml() -> List[Dict[str, Any]]:
    """Đọc danh sách nguồn backup từ sources.yaml."""
    if _SOURCES_YAML.exists():
        try:
            content = yaml.safe_load(_SOURCES_YAML.read_text(encoding="utf-8"))
            if isinstance(content, dict) and "sources" in content:
                return content["sources"]
        except Exception as e:
            log.warning("Lỗi đọc sources.yaml: %s", e)
    return []


def _fetch_content_from_url(url: str) -> Optional[str]:
    """Tải và parse nội dung sạch từ một liên kết dự phòng."""
    resp = _safe_request_get(url)
    if not resp:
        return None

    # Nếu là JSON (Wikipedia REST API)
    if "application/json" in resp.headers.get("Content-Type", "") or url.endswith(".json") or "/api/rest_v1/" in url:
        try:
            data = resp.json()
            extract = data.get("extract") or data.get("description") or ""
            if extract:
                return f"{data.get('title', '')}\n\n{extract}"
        except Exception:
            pass

    # Nếu là Markdown hoặc Text thuần
    text = resp.text
    if url.endswith(".md") or url.endswith(".txt"):
        return text[:3000]

    # Nếu là HTML: dùng BeautifulSoup trích xuất các đoạn văn bản chính
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        paragraphs = [p.get_text(strip=True) for p in soup.find_all(["p", "h2", "h3"])]
        clean = "\n".join(p for p in paragraphs if len(p) > 40)
        if len(clean) > 200:
            return clean[:3000]
    except Exception:
        pass

    return text[:2000] if len(text) > 200 else None


def _fetch_tier2_backup_list(topic: str) -> Optional[ResearchDocument]:
    """Thực thi Tầng 2: Tra cứu link backup từ sources.yaml và tải nội dung."""
    log.info("▶ [Research Tầng 2] Tra cứu danh sách link backup cho: '%s'", topic)
    sources = _load_sources_yaml()
    if not sources:
        log.warning("sources.yaml rỗng hoặc không khả dụng -> chuyển sang Tầng 3 (Offline Docs)")
        return None

    topic_lower = topic.lower()
    best_source = None
    best_score = 0

    for src in sources:
        keywords = src.get("keywords", [])
        score = sum(1 for kw in keywords if kw.lower() in topic_lower)
        if score > best_score:
            best_score = score
            best_source = src

    if not best_source or best_score == 0:
        log.warning("Không tìm thấy link backup phù hợp từ khóa -> chuyển sang Tầng 3 (Offline Docs)")
        return None

    urls = best_source.get("urls", [])
    title = best_source.get("title", best_source.get("domain", "Curated Reference"))
    log.info("✓ [Research Tầng 2] Khớp nhóm tài liệu backup: '%s' (Điểm: %d)", title, best_score)

    for url in urls:
        log.info("  -> Đang tải từ link backup: %s", url)
        content = _fetch_content_from_url(url)
        if content and len(content) > 150:
            log.info("✓ [Research Tầng 2] Tải thành công %d ký tự từ %s", len(content), url)
            return ResearchDocument(
                topic=topic,
                tier="tier2_backup_link",
                source_name=f"Backup Source ({title})",
                source_url=url,
                title=title,
                concept=content[:800],
                full_text=content,
            )

    log.warning("✗ [Research Tầng 2] Tải tất cả các link backup thất bại -> chuyển sang Tầng 3 (Offline Docs)")
    return None


# ---------------------------------------------------------------------------
# TẦNG 3: KHO TÀI LIỆU TẢI SẴN OFFLINE (Bundled Local Docs - Markdown & PDF)
# ---------------------------------------------------------------------------

def _extract_text_from_pdf(pdf_path: Path) -> str:
    """Trích xuất nội dung văn bản từ file PDF (hỗ trợ pypdf hoặc pdfminer nếu có)."""
    try:
        import pypdf
        reader = pypdf.PdfReader(str(pdf_path))
        pages_text = []
        for page in reader.pages[:15]:  # đọc tối đa 15 trang đầu
            t = page.extract_text()
            if t:
                pages_text.append(t)
        return "\n\n".join(pages_text)
    except Exception as e:
        log.debug("Trích xuất PDF %s lỗi: %s", pdf_path.name, e)
    return ""

def _extract_best_section_from_markdown(md_content: str, topic: str) -> Tuple[str, List[str], str]:
    """Trích xuất mục phù hợp nhất từ file Markdown offline cùng công thức và dữ liệu mẫu."""
    topic_words = set(re.findall(r"\w+", topic.lower()))
    sections = re.split(r"\n(?=##\s+)", md_content)
    best_sec = ""
    best_score = -1

    for sec in sections:
        sec_lower = sec.lower()
        score = sum(2 for w in topic_words if len(w) > 2 and w in sec_lower)
        # Tăng điểm nếu khớp các từ khóa then chốt
        if any(k in sec_lower for k in ("rsa", "diffie", "two pointers", "merge sort", "transformer", "cache", "fourier")):
            score += 3
        if score > best_score:
            best_score = score
            best_sec = sec

    target = best_sec if best_score > 0 else (sections[0] if sections else md_content)

    # Trích xuất công thức LaTeX / math ($...$)
    math_formulas = re.findall(r"(?:\$\$[\s\S]*?\$\$|\$[^\n$]+?\$)", target)
    cleaned_formulas = [f.strip().replace("\n", " ") for f in math_formulas[:8]]

    # Trích xuất đoạn code hoặc bảng dữ liệu mẫu
    sample_blocks = re.findall(r"(?:```[\s\S]*?```|\|[^\n]+\|(?:\n\|[^\n]+\|)+)", target)
    sample_data = "\n\n".join(b.strip() for b in sample_blocks[:2])

    return target, cleaned_formulas, sample_data


def _fetch_tier3_offline_docs(topic: str) -> ResearchDocument:
    """Thực thi Tầng 3: Sử dụng kho tài liệu chuẩn bị sẵn offline trong assets/docs/."""
    log.info("▶ [Research Tầng 3] Quét kho tài liệu chuẩn bị sẵn offline (assets/docs/) cho: '%s'", topic)

    topic_lower = topic.lower()

    # Bảng ánh xạ ưu tiên file theo từ khóa
    file_map = [
        (("rsa", "mã hóa", "mật mã", "diffie", "sha-256", "sha256", "zero-knowledge", "zkp", "qr", "reed-solomon"), "cryptography_security.md"),
        (("pointer", "sort", "sắp xếp", "merge", "quick", "b+tree", "dijkstra", "a*", "bloom", "raft", "hashing", "thuật toán"), "algorithms_dsa.md"),
        (("transformer", "attention", "q-k-v", "backprop", "gradient", "rlhf", "ppo", "diffusion", "ai", "học máy", "nơ-ron"), "ai_machine_learning.md"),
        (("cpu", "gpu", "cache", "l1", "l2", "l3", "ram", "bán dẫn", "transistor", "ssd", "hdd", "warp divergence"), "computer_architecture.md"),
        (("ray tracing", "rasterization", "shader", "sdf", "bézier", "bezier", "anti-aliasing", "đồ họa"), "graphic_programming.md"),
        (("fourier", "mandelbrot", "fractal", "game of life", "conway", "voronoi", "fibonacci", "toán"), "mathematics_simulation.md"),
        (("tcp", "ip", "gps", "wifi", "jpeg", "mp3", "nén ảnh", "thuyết tương đối"), "networking_everyday_tech.md"),
        (("python", "c++", "rust", "javascript", "v8", "jit", "goroutine", "borrow checker"), "programming_languages.md"),
    ]

    # 1. Kiểm tra nếu có file PDF phù hợp trong assets/docs/
    pdf_files = list(_DOCS_DIR.glob("*.pdf"))
    for pdf_path in pdf_files:
        words = [w for w in re.findall(r"\w+", topic_lower) if len(w) > 2]
        if any(w in pdf_path.stem.lower() for w in words):
            pdf_text = _extract_text_from_pdf(pdf_path)
            if len(pdf_text) > 100:
                section, formulas, sample_data = _extract_best_section_from_markdown(pdf_text, topic)
                log.info("✓ [Research Tầng 3] Trích xuất thành công từ file PDF offline: %s", pdf_path.name)
                return ResearchDocument(
                    topic=topic,
                    tier="tier3_offline_doc",
                    source_name=f"Tài liệu PDF offline ({pdf_path.name})",
                    source_url=f"assets/docs/{pdf_path.name}",
                    title=topic,
                    concept=section[:600],
                    math_formulas=formulas,
                    sample_data=sample_data,
                    full_text=section,
                )

    # 2. Quét file Markdown theo bảng chủ đề
    selected_filename = "algorithms_dsa.md"  # mặc định
    for keywords, fname in file_map:
        if any(kw in topic_lower for kw in keywords):
            selected_filename = fname
            break

    target_file = _DOCS_DIR / selected_filename
    if not target_file.exists():
        # Tìm bất kỳ file .md nào có sẵn trong thư mục
        md_files = list(_DOCS_DIR.glob("*.md"))
        target_file = md_files[0] if md_files else None

    if target_file and target_file.exists():
        md_text = target_file.read_text(encoding="utf-8")
        section, formulas, sample_data = _extract_best_section_from_markdown(md_text, topic)
        log.info("✓ [Research Tầng 3] Trích xuất thành công từ tài liệu offline: %s", target_file.name)
        return ResearchDocument(
            topic=topic,
            tier="tier3_offline_doc",
            source_name=f"Tài liệu chuẩn offline ({target_file.name})",
            source_url=f"assets/docs/{target_file.name}",
            title=topic,
            concept=section[:600],
            math_formulas=formulas,
            sample_data=sample_data,
            full_text=section,
        )

    # Fallback cuối cùng nếu thư mục hoàn toàn trống
    log.warning("Thư mục assets/docs/ không có file markdown, dùng dữ liệu cơ sở.")
    return ResearchDocument(
        topic=topic,
        tier="tier3_offline_doc",
        source_name="Tài liệu cốt lõi hệ thống",
        title=topic,
        concept=f"Khái niệm khoa học máy tính và thuật toán cho chủ đề: {topic}.",
        full_text=f"Thuật toán và nguyên lý hoạt động của {topic}.",
    )


# ---------------------------------------------------------------------------
# ĐIỂM VÀO DUY NHẤT (Main Entrypoint)
# ---------------------------------------------------------------------------

def gather_research(topic: str, max_chars: int = 4000) -> ResearchDocument:
    """Nghiên cứu tài liệu cho chủ đề qua 3 tầng fallback và lưu cache.
    
    1. Kiểm tra cache cục bộ (nếu có -> trả về ngay).
    2. Tầng 1: Tự động tìm & tải trên mạng (Wikipedia, DuckDuckGo, ArXiv, TinyFish).
    3. Tầng 2: Tải theo link backup uy tín trong sources.yaml.
    4. Tầng 3: Sử dụng kho tài liệu chuẩn bị sẵn offline trong assets/docs/.
    """
    topic = (topic or "").strip()
    if not topic:
        return ResearchDocument(topic="", tier="empty", source_name="None")

    # 1. Kiểm tra Cache
    cached = _load_cached_research(topic)
    if cached:
        return cached

    # 2. Tầng 1: Tự động tìm & tải trên mạng
    doc = _fetch_tier1_auto_search(topic)

    # 3. Tầng 2: Link backup
    if not doc:
        doc = _fetch_tier2_backup_list(topic)

    # 4. Tầng 3: Offline documents tải sẵn
    if not doc:
        doc = _fetch_tier3_offline_docs(topic)

    # Lưu cache và trả về
    if doc:
        _save_cached_research(doc)
    return doc
