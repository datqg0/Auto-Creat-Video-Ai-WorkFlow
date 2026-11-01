"""Script probe: test thử LLM provider trước khi sửa code.

Chạy: python test_llm.py
Cần điền GEMINI_API_KEY, GROQ_API_KEY hoặc các key khác vào .env trước.

Script sẽ:
1. Liệt kê các model Gemini còn khả dụng.
2. Test gọi Groq, Pollinations, Cerebras.
"""
from __future__ import annotations

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv

load_dotenv()


def probe_gemini() -> None:
    print("\n=== GEMINI: liệt kê model khả dụng ===")
    key = os.getenv("GEMINI_API_KEY")
    if not key or "your_" in key:
        print("  [BỎ QUA] chưa có GEMINI_API_KEY thật trong .env")
        return
    try:
        from google import genai

        client = genai.Client(api_key=key)
        for m in client.models.list():
            print("  -", m.name)
    except Exception as e:  # noqa: BLE001
        print("  LỖI:", e)


def probe_groq() -> None:
    print("\n=== GROQ: test chat.completions ===")
    key = os.getenv("GROQ_API_KEY")
    if not key or "your_" in key:
        print("  [BỎ QUA] chưa có GROQ_API_KEY thật trong .env")
        return
    try:
        from groq import Groq

        client = Groq(api_key=key)
        resp = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Nói 'xin chào' bằng tiếng Việt trong 1 câu ngắn."}],
        )
        print("  OK [openai/gpt-oss-120b]:", resp.choices[0].message.content.strip())
    except Exception as e:  # noqa: BLE001
        print("  LỖI:", e)


def probe_pollinations() -> None:
    print("\n=== POLLINATIONS (DeepSeek): test openai-compatible ===")
    key = os.getenv("POLLINATIONS_API_KEY")
    if not key or "your_" in key:
        print("  [BỎ QUA] chưa có POLLINATIONS_API_KEY thật trong .env")
        return
    try:
        from openai import OpenAI

        client = OpenAI(api_key=key, base_url="https://gen.pollinations.ai/v1")
        resp = client.chat.completions.create(
            model="deepseek/deepseek-v4.1-flash",
            max_tokens=500,
            messages=[{"role": "user", "content": "Nói 'xin chào' bằng tiếng Việt trong 1 câu ngắn."}],
        )
        print("  OK [deepseek/deepseek-v4.1-flash]:", resp.choices[0].message.content.strip())
    except Exception as e:  # noqa: BLE001
        print("  LỖI:", e)


if __name__ == "__main__":
    probe_gemini()
    probe_pollinations()
    probe_groq()
