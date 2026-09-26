"""LLM client đa tầng: thử provider theo thứ tự, lỗi thì fallback sang tầng kế.

Các tầng (cấu hình trong config.yaml -> llm.providers):
  Tier 0  Anthropic (Opus, chính)
  Tier 1  Gemini
  Tier 2  Groq (gpt-oss)
  Tier 3  OpenRouter (free models)          -> OpenAI-compatible
  Tier 3.5 Z.ai GLM Flash                    -> Anthropic-compatible (dùng lại _call_anthropic)
  Tier 4  Mistral                            -> OpenAI-compatible
  Tier 4.5 NVIDIA NIM                         -> OpenAI-compatible
  Tier 5  Cloudflare Workers AI / GitHub Models / SambaNova -> OpenAI-compatible

Mỗi provider khai báo:
  type         : anthropic | gemini | groq | openai  (bộ gọi tương ứng)
  api_key_env  : tên biến môi trường chứa API key
  base_url     : endpoint (hỗ trợ ${ENV_VAR} để chèn từ môi trường)
Trả về text. Tự thử provider kế tiếp nếu provider hiện tại lỗi.
"""
from __future__ import annotations

import logging
import os
import re
import time
from typing import Any

from .config import CONFIG, env

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


# Timeout mặc định cho MỘT request. None = không giới hạn (để Opus sinh kịch bản
# dài bao lâu cũng được, giống bản gốc). Có thể override qua provider['timeout'].
_DEFAULT_TIMEOUT: float | None = None
# Giới hạn thời gian tối đa chịu chờ khi 429 báo retry_delay; dài hơn thì bỏ qua
# provider để chuyển sang cái kế cho nhanh. Gemini free-tier thường báo ~20-40s,
# nên để đủ rộng để chờ hết cửa sổ rate-limit thay vì bỏ luôn provider cuối.
_MAX_RETRY_WAIT = 65.0


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


def _api_key(provider: dict, default_env: str) -> str:
    key_env = provider.get("api_key_env", default_env)
    val = env(key_env)
    if not val:
        raise LLMError(f"Thiếu {key_env}")
    return val


def _resolve_base_url(url: str | None) -> str | None:
    """Chèn ${ENV_VAR} trong base_url từ môi trường (vd Cloudflare account id)."""
    if not url:
        return url
    return re.sub(r"\$\{(\w+)\}", lambda m: os.getenv(m.group(1), ""), url)


def _call_gemini(provider: dict, prompt: str, system: str, temperature: float) -> str:
    import google.generativeai as genai

    api_key = _api_key(provider, "GEMINI_API_KEY")
    genai.configure(api_key=api_key)
    gm = genai.GenerativeModel(provider["model"], system_instruction=system or None)
    timeout = provider.get("timeout", _DEFAULT_TIMEOUT)
    req_opts = {"timeout": float(timeout)} if timeout is not None else {}
    resp = gm.generate_content(
        prompt,
        generation_config={"temperature": temperature},
        request_options=req_opts,
    )
    text = (resp.text or "").strip()
    if not text:
        raise LLMError("Gemini trả về rỗng")
    return text


def _call_anthropic(provider: dict, prompt: str, system: str, temperature: float) -> str:
    from anthropic import Anthropic

    api_key = _api_key(provider, "ANTHROPIC_API_KEY")
    kwargs: dict = {
        "api_key": api_key,
        # Tắt retry nội bộ của SDK; generate() tự quản lý retry/fallback.
        "max_retries": 0,
    }
    timeout = provider.get("timeout", _DEFAULT_TIMEOUT)
    if timeout is not None:
        kwargs["timeout"] = float(timeout)
    base_url = _resolve_base_url(provider.get("base_url"))
    if base_url:
        kwargs["base_url"] = base_url
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


def _call_groq(provider: dict, prompt: str, system: str, temperature: float) -> str:
    from groq import Groq

    api_key = _api_key(provider, "GROQ_API_KEY")
    kwargs: dict = {"api_key": api_key}
    timeout = provider.get("timeout", _DEFAULT_TIMEOUT)
    if timeout is not None:
        kwargs["timeout"] = float(timeout)
    base_url = _resolve_base_url(provider.get("base_url"))
    if base_url:
        kwargs["base_url"] = base_url
    client = Groq(**kwargs)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=provider["model"],
        max_tokens=int(provider.get("max_output_tokens", 8192)),
        temperature=temperature,
        messages=messages,
    )
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise LLMError("Groq trả về rỗng")
    return text


def _call_openai(provider: dict, prompt: str, system: str, temperature: float) -> str:
    """Bộ gọi chung cho mọi endpoint tương thích OpenAI (OpenRouter, Mistral,
    NVIDIA NIM, GitHub Models, SambaNova, Cloudflare Workers AI...)."""
    from openai import OpenAI

    api_key = _api_key(provider, "OPENAI_API_KEY")
    kwargs: dict = {"api_key": api_key}
    base_url = _resolve_base_url(provider.get("base_url"))
    if base_url:
        kwargs["base_url"] = base_url
    timeout = provider.get("timeout", _DEFAULT_TIMEOUT)
    if timeout is not None:
        kwargs["timeout"] = float(timeout)
    # OpenRouter khuyến nghị 2 header nhận diện; vô hại với provider khác.
    default_headers = provider.get("headers") or None
    if default_headers:
        kwargs["default_headers"] = default_headers
    client = OpenAI(**kwargs)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    create_kw: dict = {
        "model": provider["model"],
        "max_tokens": int(provider.get("max_output_tokens", 8192)),
        "messages": messages,
    }
    if provider.get("supports_temperature", True):
        create_kw["temperature"] = temperature
    resp = client.chat.completions.create(**create_kw)
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise LLMError(f"{provider['name']} trả về rỗng")
    return text


_DISPATCH = {
    "gemini": _call_gemini,
    "anthropic": _call_anthropic,
    "groq": _call_groq,
    "openai": _call_openai,
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
    default_retries = int(llm_cfg.get("max_retries", 2))

    last_err: Exception | None = None
    for provider in llm_cfg["providers"]:
        name = provider["name"]
        # 'type' chọn bộ gọi; mặc định suy ra từ name để tương thích cấu hình cũ.
        kind = provider.get("type", name)
        fn = _DISPATCH.get(kind)
        if not fn:
            log.warning("Provider không hỗ trợ: %s (type=%s)", name, kind)
            continue
        # Số lần thử của riêng provider (vd Opus 3 lần) rồi mới fallback.
        retries = int(provider.get("retries", default_retries))
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
