"""LLM client với rotation: Gemini Flash (chính) -> Opus (fallback).

Trả về text. Tự thử provider kế tiếp nếu provider hiện tại lỗi.
"""
from __future__ import annotations

import logging
from typing import Any

from .config import CONFIG, env

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


def _call_gemini(provider: dict, prompt: str, system: str, temperature: float) -> str:
    import google.generativeai as genai

    api_key = env("GEMINI_API_KEY")
    if not api_key:
        raise LLMError("Thiếu GEMINI_API_KEY")
    genai.configure(api_key=api_key)
    gm = genai.GenerativeModel(provider["model"], system_instruction=system or None)
    resp = gm.generate_content(
        prompt,
        generation_config={"temperature": temperature},
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
    kwargs: dict = {"api_key": api_key}
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
    """Gọi LLM theo thứ tự provider trong config, fallback khi lỗi."""
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
    raise LLMError(f"Tất cả LLM provider đều lỗi. Cuối: {last_err}")
