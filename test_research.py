"""Kiểm thử hoạt động của hệ thống nghiên cứu tài liệu 3 tầng fallback."""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import logging
from src.research_engine import (
    gather_research,
    _fetch_tier1_auto_search,
    _fetch_tier2_backup_list,
    _fetch_tier3_offline_docs,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")

print("\n" + "=" * 60)
print("TEST 1: Tầng 1 - Tự động tìm & tải trên mạng (Wikipedia / DDG / ArXiv)")
print("=" * 60)
doc_tier1 = _fetch_tier1_auto_search("Mật mã RSA")
if doc_tier1:
    print(f"✓ Tier 1 thành công: [{doc_tier1.source_name}]")
    print(f"  Title: {doc_tier1.title}")
    print(f"  Concept ({len(doc_tier1.concept)} chars): {doc_tier1.concept[:160]}...")
else:
    print("✗ Tier 1 không tìm thấy")

print("\n" + "=" * 60)
print("TEST 2: Tầng 2 - Tải theo link backup (Curated sources.yaml)")
print("=" * 60)
doc_tier2 = _fetch_tier2_backup_list("Bloom Filter")
if doc_tier2:
    print(f"✓ Tier 2 thành công: [{doc_tier2.source_name}]")
    print(f"  URL: {doc_tier2.source_url}")
    print(f"  Snippet ({len(doc_tier2.full_text)} chars): {doc_tier2.full_text[:160]}...")
else:
    print("✗ Tier 2 không tìm thấy")

print("\n" + "=" * 60)
print("TEST 3: Tầng 3 - Kho tài liệu chuẩn bị sẵn offline (assets/docs/*.md)")
print("=" * 60)
doc_tier3 = _fetch_tier3_offline_docs("Thuật toán mã hóa RSA và số nguyên tố")
if doc_tier3:
    print(f"✓ Tier 3 thành công: [{doc_tier3.source_name}]")
    print(f"  File: {doc_tier3.source_url}")
    print(f"  Công thức tìm thấy ({len(doc_tier3.math_formulas)}): {doc_tier3.math_formulas[:3]}")
    if doc_tier3.sample_data:
        print(f"  Dữ liệu mẫu:\n{doc_tier3.sample_data[:200]}")
else:
    print("✗ Tier 3 không tìm thấy")

print("\n" + "=" * 60)
print("TEST 4: Điểm vào toàn diện gather_research() & to_prompt_block()")
print("=" * 60)
doc_full = gather_research("Self-Attention trong Transformer")
print(f"Kết quả: Tier={doc_full.tier}, Source={doc_full.source_name}")
prompt_block = doc_full.to_prompt_block(max_chars=1200)
print("\n--- MẪU PROMPT BLOCK SẼ BẮT BUỘC LLM & ANIMATION TUÂN THEO: ---")
print(prompt_block)
print("\n✓ Hoàn tất kiểm thử Research Engine!")
