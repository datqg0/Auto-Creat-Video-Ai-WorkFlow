"""LLM client với rotation: Gemini Flash (chính) -> Opus (fallback).

Trả về text. Tự thử provider kế tiếp nếu provider hiện tại lỗi.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any

from .config import CONFIG, env

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


# Timeout mặc định cho MỘT request (giây). Provider proxy hay treo (Cloudflare 524)
# nên phải chặn cứng để fail nhanh rồi fallback thay vì đợi 120s×N của SDK.
_DEFAULT_TIMEOUT = 60
# Giới hạn thời gian tối đa chịu chờ khi 429 báo retry_delay; dài hơn thì bỏ qua
# provider để chuyển sang cái kế cho nhanh.
_MAX_RETRY_WAIT = 20.0


def _retry_after(e: Exception) -> float | None:
    """Rút retry_delay/retry_after (giây) từ exception 429 nếu có."""
    # Thuộc tính chuẩn (một số SDK gắn kèm)
    for attr in ("retry_after", "retry_delay"):
        val = getattr(e, attr, None)
        if val is None:
            continue
        secs = getattr(val, "seconds", None)
        if secs is not None:
            return float(secs)
        try:
            return float(val)
        except (TypeError, ValueError):
            pass
    # Fallback: dò trong message ("retry_delay { seconds: 40 }" hoặc "retry after 40s")
    msg = str(e)
    m = re.search(r"seconds:\s*(\d+)", msg) or re.search(r"retry[_\s-]?after[:\s]*(\d+)", msg, re.I)
    if m:
        return float(m.group(1))
    return None


def _is_rate_limit(e: Exception) -> bool:
    name = type(e).__name__.lower()
    msg = str(e).lower()
    return "ratelimit" in name or "resourceexhausted" in name or "429" in msg or "quota" in msg


def _call_gemini(provider: dict, prompt: str, system: str, temperature: float) -> str:
    import google.generativeai as genai

    api_key = env("GEMINI_API_KEY")
    if not api_key:
        raise LLMError("Thiếu GEMINI_API_KEY")
    genai.configure(api_key=api_key)
    gm = genai.GenerativeModel(provider["model"], system_instruction=system or None)
    timeout = float(provider.get("timeout", _DEFAULT_TIMEOUT))
    resp = gm.generate_content(
        prompt,
        generation_config={"temperature": temperature},
        request_options={"timeout": timeout},
    )
    text = (resp.text or "").strip()
    if not text:
        raise LLMError("Gemini trả về rỗng")
    return text


def _call_anthropic(provider: dict, prompt: str, system: str, temperature: float) -> str:
    from anthropic import Anthropic

    api_key = env("ANTHROPIC_API_KEY")
    if not api_key:
        raise LLMError("Thiếu ANTHROPIC_API_KEY")
    kwargs: dict = {
        "api_key": api_key,
        # Chặn cứng timeout để 524 fail nhanh thay vì treo tới read-timeout của proxy.
        "timeout": float(provider.get("timeout", _DEFAULT_TIMEOUT)),
        # Tắt retry nội bộ của SDK (mặc định 2 lần, mỗi lần backoff ~ tới 120s).
        # generate() bên dưới tự quản lý retry/fallback nên không cần SDK nhân đôi.
        "max_retries": 0,
    }
    if provider.get("base_url"):
        kwargs["base_url"] = provider["base_url"]
    client = Anthropic(**kwargs)
    kw: dict = {
        "model": provider["model"],
        "max_tokens": int(provider.get("max_output_tokens", 4096)),
        "system": system or "",
        "messages": [{"role": "user", "content": prompt}],
    }
    # provider justwoker.icu không nhận 'temperature'; chỉ gửi khi được bật
    if provider.get("supports_temperature", False):
        kw["temperature"] = temperature
    resp = client.messages.create(**kw)
    parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
    text = "".join(parts).strip()
    if not text:
        raise LLMError("Anthropic trả về rỗng")
    return text


_DISPATCH = {
    "gemini": _call_gemini,
    "anthropic": _call_anthropic,
}


def generate(prompt: str, system: str = "") -> str:
    """Gọi LLM theo thứ tự provider trong config, fallback khi lỗi.

    - Timeout cứng mỗi request (xem _call_*): tránh treo lâu vì proxy 524.
    - 429 (rate limit): nếu retry_delay ngắn thì đợi rồi thử lại 1 lần trên
      cùng provider; nếu dài hơn _MAX_RETRY_WAIT thì bỏ qua sang provider kế.
    - Các lỗi khác: backoff ngắn có jitter rồi thử lại theo max_retries.
    """
    import random

    llm_cfg = CONFIG["llm"]
    temperature = float(llm_cfg.get("temperature", 0.9))
    retries = int(llm_cfg.get("max_retries", 2))

    last_err: Exception | None = None
    for provider in llm_cfg["providers"]:
        name = provider["name"]
        fn = _DISPATCH.get(name)
        if not fn:
            log.warning("Provider không hỗ trợ: %s", name)
            continue
        for attempt in range(1, retries + 1):
            try:
                log.info("LLM %s (%s) attempt %d", name, provider["model"], attempt)
                return fn(provider, prompt, system, temperature)
            except Exception as e:  # noqa: BLE001 - muốn fallback mọi lỗi
                last_err = e
                log.warning("LLM %s lỗi (attempt %d): %s", name, attempt, e)

                is_last_attempt = attempt >= retries
                if _is_rate_limit(e):
                    wait = _retry_after(e)
                    # Rate limit + delay quá dài -> bỏ provider này, sang cái kế.
                    if wait is None or wait > _MAX_RETRY_WAIT:
                        log.warning(
                            "LLM %s rate-limited (delay=%s) -> chuyển provider",
                            name, wait,
                        )
                        break
                    if not is_last_attempt:
                        sleep_s = wait + random.uniform(0.2, 1.0)
                        log.info("LLM %s chờ %.1fs rồi thử lại", name, sleep_s)
                        time.sleep(sleep_s)
                        continue
                    break
                # Lỗi khác: backoff ngắn có jitter rồi thử lại.
                if not is_last_attempt:
                    sleep_s = min(2.0 * attempt, 5.0) + random.uniform(0.1, 0.5)
                    time.sleep(sleep_s)
    raise LLMError(f"Tất cả LLM provider đều lỗi. Cuối: {last_err}")
