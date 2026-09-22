"""Script probe: test thử LLM provider trước khi sửa code.

Chạy: python test_llm.py
Cần điền GEMINI_API_KEY và ANTHROPIC_API_KEY vào .env trước.

Script sẽ:
1. Liệt kê các model Gemini còn khả dụng.
2. Test gọi Opus qua provider justwoker.icu, thử CÓ và KHÔNG có temperature.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

OPUS_BASE_URL = "https://api.justwoker.icu"
OPUS_MODEL = "claude-opus-4-8"


def probe_gemini() -> None:
    print("\n=== GEMINI: liệt kê model khả dụng ===")
    key = os.getenv("GEMINI_API_KEY")
    if not key or "your_" in key:
        print("  [BỎ QUA] chưa có GEMINI_API_KEY thật trong .env")
        return
    try:
        import google.generativeai as genai

        genai.configure(api_key=key)
        for m in genai.list_models():
            if "generateContent" in getattr(m, "supported_generation_methods", []):
                print("  -", m.name)
    except Exception as e:  # noqa: BLE001
        print("  LỖI:", e)


def probe_opus() -> None:
    print("\n=== OPUS (justwoker.icu): test messages.create ===")
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key or "your_" in key:
        print("  [BỎ QUA] chưa có ANTHROPIC_API_KEY thật trong .env")
        return
    from anthropic import Anthropic

    client = Anthropic(api_key=key, base_url=OPUS_BASE_URL)

    # Thử CÓ temperature (giống code hiện tại)
    print("\n  -> Thử CÓ temperature:")
    try:
        resp = client.messages.create(
            model=OPUS_MODEL,
            max_tokens=64,
            temperature=0.9,
            messages=[{"role": "user", "content": "Nói 'xin chào' bằng 1 từ."}],
        )
        print("     OK:", "".join(b.text for b in resp.content if getattr(b, "type", None) == "text"))
    except Exception as e:  # noqa: BLE001
        print("     LỖI:", e)

    # Thử KHÔNG temperature
    print("\n  -> Thử KHÔNG temperature:")
    try:
        resp = client.messages.create(
            model=OPUS_MODEL,
            max_tokens=64,
            messages=[{"role": "user", "content": "Nói 'xin chào' bằng 1 từ."}],
        )
        print("     OK:", "".join(b.text for b in resp.content if getattr(b, "type", None) == "text"))
    except Exception as e:  # noqa: BLE001
        print("     LỖI:", e)


if __name__ == "__main__":
    probe_gemini()
    probe_opus()
