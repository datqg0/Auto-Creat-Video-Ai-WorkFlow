# DEV.md — Kiểm Tra & Thống Kê Dự Án
**Auto-Create-Video-AI-WorkFlow** | Cập nhật: 2026-10-13

---

## 1. Tổng Quan Codebase

| Thống kê | Giá trị |
|---|---|
| Tổng file Python (`src/`) | 28 file |
| File lớn nhất | `motion_graphics.py` (~213 KB, 4236 dòng) |
| File phức tạp nhất | `script_writer.py` (~91 KB, 1709 dòng) |
| Tổng file tài liệu | README.md, ROADMAP.md, Architecture.md, fix.md, sub.md, ANIMATION_UPGRADE.md |
| Python version target | 3.11 |
| CI/CD | GitHub Actions (cron 2×/ngày + manual trigger) |

### Sơ đồ pipeline tổng quát

```
topic_selector.py
    ↓ (LLM + DB dedup)
script_writer.py
    ↓ (JSON: scenes, narration, visual specs)
pipeline.py  <── series_manager.py (Series Engine)
    ├── tts.py              (VieNeu → Edge-TTS)
    ├── ai_code_runner.py   (Manim → Matplotlib → MathViz)
    ├── motion_graphics.py  (HTML5/Edge headless render)
    ├── visual_engine.py    (Pillow/Matplotlib PNG)
    ├── image_fetcher.py    (Google → DuckDuckGo → Pexels → FLUX)
    ├── ai_video.py         (ai_motion / Fal.ai)
    └── compositor.py       (MoviePy + FFmpeg)
            ↓
        subtitles.py → thumbnail_ai.py → youtube_uploader.py
```

---

## 2. Thống Kê Lỗi Đã Phát Hiện

### 2.1 Lỗi từ `fix.md` và Bug mới — ĐÃ FIX trong code thực tế

| Mã | Mô tả | Xác nhận trong code |
|---|---|---|
| FIX-01 | MoviePy frame EOF: thêm extra_args vào ani.save() | Đã có trong _PRELUDE dòng 82 và prompt dòng 259 |
| FIX-02 | _MANIM_QUALITY_FLAG đặt sau hàm dùng nó | Đã di chuyển lên dòng 526, trước run_manim_code() |
| FIX-03 | TTS Short bị cắt audio giữa câu | Đã thêm audio fade-out 0.5s tại compositor.py dòng 415–417 |
| FIX-04 | Animation Matplotlib quá nhanh | Đã có rule nhịp chậm trong prompt generate_matplotlib_from_prompt() |
| FIX-05 | Animation Manim quá nhanh | Đã có rule nhịp trong generate_manim_from_prompt() |
| FIX-08 | min_words Short mode (DEV-01) | Đã tách accept_words_ratio trong _Settings: 0.80 (short), 0.75 (long) |
| FIX-09 | Rule nhịp chậm Short prompt (DEV-02) | Đã thêm rule visual_prompt dừng ≥0.8s, 3–4 bước vào _build_short_prompt() |
| FIX-10 | W/H cache tại import time trong compositor.py | Đã đổi thành _W(), _H(), _FPS() (dòng 45–54) |
| FIX-11 | W/H cache tại import time trong visual_engine.py | Đã đổi thành _W(), _H() (dòng 31–36) |
| FIX-12 | Timeout Matplotlib (DEV-03) | Đã đổi thành dynamic timeout max(150, int(duration * fps / 35)) trong run_ai_code() |
| FIX-13 | Manim: stderr chưa định nghĩa khi proc=None | Đã gán stderr = "" trước vòng lặp (dòng 557) |
| FIX-18 | json_repair thiếu trong requirements.txt | Đã có json-repair>=0.30.0 (dòng 44) |
| BUG-NEW-01 | _XFADE module-level (DEV-04) | Đã đổi thành hàm _xfade() dynamic trong compositor.py |
| BUG-NEW-02 | Reset _DISABLED_PROVIDERS thiếu backoff (DEV-06) | Đã thêm time.sleep(2.0) và cảnh báo trước khi clear provider trong llm.py |
| BUG-NEW-03 | Topic selector prompt khi language=en | Đã thêm template prompt và hint tiếng Anh trong topic_selector.py |
| BUG-NEW-04 | _write_two_part_script thiếu fallback exercise | Đã thêm fallback tự động gọi _generate_exercises() trong script_writer.py |
| PI-01 | Migrate google.generativeai → google.genai (DEV-05) | Đã chuyển sang google.genai Client SDK mới nhất trong llm.py |

### 2.2 Lỗi từ `fix.md` — CHƯA FIX trong code

| Mã | Ưu tiên | File | Mô tả chi tiết |
|---|---|---|---|
| FIX-16 | HIGH | script_writer.py | Theo fix.md: _clean_narration_markdown() dùng sai biến text thay vì t. Tuy nhiên, function này KHÔNG TỒN TẠI trong code hiện tại. Code thực tế dùng _clean_narration() + _clean_display() và chuỗi biến đúng. |
| FIX-06 | MEDIUM | script_writer.py | Prompt _build_long_prompt() chưa có ví dụ mẫu bảng ĐÚNG/SAI với dữ liệu số cụ thể. |
| FIX-07 | MEDIUM | script_writer.py | visual_prompt mẫu thiếu ví dụ đầy đủ 3 pha (data → phase1 → phase2 → phase3) với timing. |
| FIX-14 | LOW | script_writer.py | System prompt thiếu ví dụ ẩn dụ Feynman cụ thể (Hash Table, TCP/IP, Mutex). |
| FIX-15 | LOW | .github/workflows/ | Workflow không upload artifact khi pipeline lỗi, khó debug CI crash. |
| FIX-17 | LOW | config.yaml | VieNeu speed: 0.95 vs Edge rate: -6% chưa đồng nhất; nhịp đọc thay đổi khi fallback. |

---

### 2.3 Bug mới phát hiện — Tình trạng cập nhật

- **BUG-NEW-01** (compositor.py: _XFADE module-level): **[ĐÃ FIX]** Chuyển sang hàm `_xfade()` dynamic.
- **BUG-NEW-02** (llm.py: Reset _DISABLED_PROVIDERS thiếu backoff): **[ĐÃ FIX]** Thêm sleep 2.0s và cảnh báo log trước khi reset.
- **BUG-NEW-03** (topic_selector.py: Prompt khi language=en): **[ĐÃ FIX]** Template prompt và hint tự động chuyển sang tiếng Anh chuẩn khi language=en.
- **BUG-NEW-04** (script_writer.py: _write_two_part_script thiếu fallback exercise): **[ĐÃ FIX]** Tự động fallback sang `_generate_exercises()` nếu Part 2 không sinh bài tập.
- **BUG-NEW-05** (motion_graphics.py: File 4236 dòng): *Kỹ thuật nợ dài hạn* (DEV-12).

---

### 2.4 Vấn đề tiềm ẩn (Potential Issues)

| ID | File | Mô tả | Tình trạng |
|---|---|---|---|
| PI-01 | llm.py | google.generativeai deprecated | **[ĐÃ FIX]** Migrate sang google.genai Client |
| PI-02 | requirements.txt | moviepy>=1.0.3,<2.0.0 ghim 1.x nhưng mv_compat.py đã hỗ trợ 2.x | LOW |
| PI-03 | ai_code_runner.py | Sandbox subprocess không giới hạn memory/CPU, chỉ timeout | MEDIUM — CI có thể bị OOM video dài |
| PI-04 | db.py | Neon free-tier tự ngủ sau 5 ngày → fallback SQLite | MEDIUM |
| PI-05 | motion_graphics.py | Headless browser (Edge/Chrome) không có trên runner | MEDIUM |
| PI-06 | Toàn bộ | Không có test tự động | HIGH (kỹ thuật nợ) |

---

## 3. Đánh Giá Chất Lượng Code

### Điểm mạnh

- **Fallback phân tầng xuất sắc**: LLM (10 tier), TTS (3 tier), Image (5 tier), DB (4 tier), Animation (4 tier). Rất hiếm hệ thống tự động nào có nhiều lớp dự phòng như vậy.
- **Bảo mật tốt**: Không có eval/exec code LLM. Code AI chạy trong subprocess riêng với env sạch.
- **Xử lý lỗi nhất quán**: Hầu hết mọi chỗ có try/except với fallback, pipeline không crash.
- **Config tập trung**: Tất cả hành vi tuỳ chỉnh qua config.yaml, không cần sửa code.
- **Prompt chất lượng cao**: Hệ thống prompt kỹ, có triết lý sư phạm, ví dụ, rule ràng buộc JSON.
- **FIX-10/FIX-11 đã áp dụng đúng**: _W()/_H()/_FPS() dynamic là fix kiến trúc quan trọng.

### Điểm yếu

- **Thiếu test hoàn toàn**: Không có test tự động; regression chỉ phát hiện khi chạy thực tế.
- **motion_graphics.py quá lớn**: 4236 dòng, 213 KB — khó navigate, review và debug.
- **Prompt dài**: Một số LLM free-tier bị TPM limit với prompt lớn.
- **topic_selector.py dùng tiếng Việt không dấu**: Khó đọc, inconsistent với các file khác.

---

## 4. Hướng Phát Triển (Prioritized)

### 🔴 Ưu Tiên Cao (Fix ngay, ảnh hưởng chất lượng video)

#### DEV-01 — Nâng ngưỡng min_words cho Short mode (FIX-08) [ĐÃ HOÀN THÀNH]
File: script_writer.py
- Đã tách accept_words_ratio trong `_Settings`: 0.80 (short), 0.75 (long/mega).
- Short 175s: min_words đạt 303 từ → đủ ~140s @ 130 wpm + pause.

#### DEV-02 — Thêm rule nhịp chậm vào Short prompt (FIX-09) [ĐÃ HOÀN THÀNH]
File: script_writer.py → _build_short_prompt()
- Đã thêm: "NHỊP CHẬM ANIMATION: Mỗi scene animation PHẢI có visual_prompt mô tả nhịp CHẬM RÃI: mỗi bước dừng ≥0.8s. Tối đa 3–4 bước."

#### DEV-03 — Tăng timeout Matplotlib (FIX-12) [ĐÃ HOÀN THÀNH]
File: ai_code_runner.py
- Đã chuyển sang timeout động: `actual_timeout = timeout if (timeout is not None and timeout > 0) else max(150, int(duration * fps / 35))`.

#### DEV-04 — _XFADE module-level (BUG-NEW-01) [ĐÃ HOÀN THÀNH]
File: compositor.py
- Đã đổi `_XFADE` thành hàm `def _xfade() -> float: return float(CONFIG["visual"].get("crossfade", 0.4))`.
- Cập nhật toàn bộ các điểm sử dụng `_XFADE` sang `xf = _xfade()` động.

---

### 🟡 Ưu Tiên Trung Bình (Cải thiện độ tin cậy & chất lượng)

#### DEV-05 — Migrate google.generativeai → google.genai (PI-01) [ĐÃ HOÀN THÀNH]
File: llm.py → _call_gemini()
- Đã thay thế thư viện cũ bằng `from google import genai; from google.genai import types`.
- Khởi tạo `genai.Client(api_key=..., http_options=...)` và gọi `client.models.generate_content(...)`.

#### DEV-06 — Thêm backoff khi reset _DISABLED_PROVIDERS (BUG-NEW-02) [ĐÃ HOÀN THÀNH]
File: llm.py
- Đã thêm `time.sleep(2.0)` và log cảnh báo khi toàn bộ provider bị vô hiệu hoá trước khi clear & retry.

#### DEV-07 — Bổ sung ví dụ mẫu cụ thể vào prompt (FIX-06/07)
File: script_writer.py
- Thêm bảng ĐÚNG/SAI với số liệu cụ thể vào _PEDAGOGY.
- Thêm ví dụ visual_prompt đầy đủ 3 pha có timing vào _scene_example().

#### DEV-08 — Ví dụ ẩn dụ Feynman vào system prompt (FIX-14)
File: script_writer.py → _SYSTEM_VI / _SYSTEM_EN
- Thêm ≤5 dòng: Hash Table = thủ thư đánh nhãn ngăn kéo; TCP = gói bưu phẩm số thứ tự; Mutex = chìa khoá nhà vệ sinh.

#### DEV-09 — Upload debug artifact khi pipeline lỗi (FIX-15)
File: .github/workflows/create-video.yml
- Thêm step upload-artifact với if: failure() để debug CI crash.

#### DEV-10 — Đồng nhất TTS speed (FIX-17)
File: config.yaml
- Đổi VieNeu speed: 0.90 và Edge rate: -10%.

---

### 🟢 Ưu Tiên Thấp / Dài Hạn (Cải thiện kiến trúc)

#### DEV-11 — Viết test tự động (kỹ thuật nợ cao nhất)
- test_script_writer.py: mock LLM, kiểm tra _check_script, _normalize_scene_dict, _finalize_title.
- test_llm.py: mock provider, kiểm tra fallback, 429 handling, quota skip.
- Smoke test pipeline.py --dry-run trong CI.

#### DEV-12 — Tách motion_graphics.py thành nhiều file
- motion_graphics/core.py — runner, browser, FFmpeg logic.
- motion_graphics/templates/ — mỗi template là 1 file .html hoặc .py riêng.

#### DEV-13 — Cache kịch bản đã sinh
File: pipeline.py / script_writer.py
- Lưu JSON kịch bản vào workdir/script.json.
- Khi retry pipeline, load lại thay vì gọi LLM mới.

#### DEV-14 — Song song hoá render scene
File: pipeline.py → _render_video()
- TTS cho tất cả scene chạy song song với ThreadPoolExecutor.
- Image fetch song song với TTS; animation/render vẫn tuần tự.

#### DEV-15 — Dashboard theo dõi pipeline
- Web UI (Flask/FastAPI) hoặc Telegram bot thông báo sau mỗi lần chạy.

#### DEV-16 — A/B test thumbnail & tiêu đề
- Thử 2 style mỗi video, track CTR qua YouTube Analytics API.

#### DEV-17 — Hỗ trợ đầy đủ tiếng Anh (BUG-NEW-03 đã giải quyết topic_selector)
- Kiểm tra end-to-end với language: en; dịch toàn bộ prompt pipeline còn lại.

---

## 5. Ma Trận Ưu Tiên Tổng Hợp

| ID | Mô tả | Ưu tiên | Trạng thái | File chính |
|---|---|---|---|---|
| DEV-01 | Nâng min_words Short | 🔴 HIGH | ✅ ĐÃ XONG | script_writer.py |
| DEV-02 | Rule nhịp Short prompt | 🔴 HIGH | ✅ ĐÃ XONG | script_writer.py |
| DEV-03 | Tăng Matplotlib timeout | 🔴 HIGH | ✅ ĐÃ XONG | ai_code_runner.py |
| DEV-04 | _XFADE dynamic | 🔴 HIGH | ✅ ĐÃ XONG | compositor.py |
| DEV-05 | Migrate google.genai | 🟡 MEDIUM | ✅ ĐÃ XONG | llm.py |
| DEV-06 | Backoff reset disabled providers | 🟡 MEDIUM | ✅ ĐÃ XONG | llm.py |
| DEV-07 | Ví dụ cụ thể vào prompt | 🟡 MEDIUM | Chưa làm | script_writer.py |
| DEV-08 | Ví dụ Feynman system prompt | 🟡 MEDIUM | Chưa làm | script_writer.py |
| DEV-09 | Debug artifact khi fail | 🟡 MEDIUM | Chưa làm | create-video.yml |
| DEV-10 | Đồng nhất TTS speed | 🟡 MEDIUM | Chưa làm | config.yaml |
| DEV-11 | Test tự động | 🟢 LONG | Chưa làm | test/ |
| DEV-12 | Tách motion_graphics.py | 🟢 LONG | Chưa làm | motion_graphics/ |
| DEV-13 | Cache kịch bản LLM | 🟢 LONG | Chưa làm | pipeline.py |
| DEV-14 | Song song hoá render | 🟢 LONG | Chưa làm | pipeline.py |
| DEV-15 | Dashboard theo dõi | 🟢 LONG | Chưa làm | mới |
| DEV-16 | A/B test thumbnail | 🟢 LONG | Chưa làm | thumbnail_ai.py |
| DEV-17 | Full English support | 🟢 LONG | Đang làm | nhiều file |

---

## 6. Ghi Chú Audit

### Sự khác biệt fix.md vs code thực tế

- **FIX-16**: function _clean_narration_markdown() không tồn tại trong code. Hàm thực tế _clean_narration() gọi _clean_display() với chuỗi biến đúng — không thấy bug text vs t.
- **FIX-01**: Đã fix trong `_PRELUDE` và prompt.
- **FIX-03**: Đã fix tại compositor.py dòng 415–417.
- **FIX-10/FIX-11/BUG-NEW-01**: Đã hoàn toàn dynamic hoá `_W()`, `_H()`, `_FPS()`, và `_xfade()`. Không còn rủi ro cache biến visual ở import time.

---

*Kiểm tra lần: 2026-10-13 | Người kiểm tra: AI audit. Cập nhật sau mỗi sprint lớn.*
