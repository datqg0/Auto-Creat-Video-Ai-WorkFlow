# Hướng Phát Triển (Roadmap)

Tài liệu định hướng phát triển cho hệ thống tự động tạo & đăng video YouTube tiếng Việt
(phong cách Kurzgesagt / 3Blue1Brown). Cập nhật theo tình trạng thực tế của codebase.

---

## 1. Tổng Quan Kiến Trúc Hiện Tại

Pipeline chạy tuần tự trong `src/pipeline.py` (`run_once`):

```
topic_selector → script_writer (LLM) → tts → visual_engine / mathviz
   → compositor (ghép + burn phụ đề) → thumbnail_ai → youtube_uploader
```

| Module | Vai trò | Ghi chú |
| --- | --- | --- |
| `topic_selector.py` | Chọn chủ đề | |
| `script_writer.py` | Sinh kịch bản JSON qua LLM | Tiêu đề = câu hỏi, Title Case |
| `llm.py` | Client LLM đa provider + fallback | Opus → Gemini → Groq |
| `tts.py` | Tổng hợp giọng đọc | VieNeu-TTS / ElevenLabs / edge-tts |
| `visual_engine.py` | Render ảnh scene + overlay | Text đặt trên vùng phụ đề |
| `mathviz/` | Thư viện animation tự viết | Không dùng Manim/LaTeX |
| `animation_bridge.py` | Cầu nối spec → mathviz | Có preset `custom` (LLM tự thiết kế) |
| `compositor.py` | Ghép video + burn phụ đề | |
| `thumbnail_ai.py` | Sinh thumbnail | FLUX / Gemini image |
| `youtube_uploader.py` | Upload | Scope `youtube.upload` |

---

## 2. Đã Hoàn Thành

- [x] **Đồng bộ A/V** — sửa lệch "giọng đi trước hình".
- [x] **Tiêu đề dạng câu hỏi** — mỗi video có title là 1 câu hỏi (`_as_question`).
- [x] **Tiêu đề Title Case** — viết hoa chữ cái đầu mỗi từ, giữ acronym (`_title_case`).
- [x] **Animation LLM tự thiết kế** — preset `custom` diễn giải spec khai báo, KHÔNG exec code.
- [x] **Text không đè phụ đề** — chữ trên màn hình đặt trên vùng phụ đề (~18% đáy).
- [x] **LLM resilience** — timeout cấu hình được, retry theo provider, xử lý 429, 3 tầng
      fallback (Opus → Gemini → Groq).

---

## 3. Ưu Tiên Ngắn Hạn

### 3.1 Ổn định LLM
- [ ] Thêm `GROQ_API_KEY` vào GitHub Secrets + truyền vào env trong workflow.
- [ ] Cân nhắc nâng cấp `google.generativeai` (đã deprecated) → `google.genai`.
- [ ] Cache kịch bản đã sinh để tránh gọi lại LLM khi retry pipeline.

### 3.2 Chất lượng nội dung
- [ ] Kiểm tra chất lượng tiếng Việt của Groq (llama-3.3-70b) khi là tầng cuối.
- [ ] Kiểm định title không vượt 100 ký tự sau khi Title Case.
- [ ] Validate JSON kịch bản chặt hơn (schema) trước khi render.

### 3.3 Dọn dẹp
- [ ] Xóa các file test tạm trong `output/preview/*.png`.
- [ ] Gộp/điền bảng trống trong `Architecture.md`.

---

## 4. Ưu Tiên Trung Hạn

### 4.1 Animation phong phú hơn
- [ ] Bổ sung object types cho preset `custom` (đường cong tham số, vùng tô, nhãn động).
- [ ] Thư viện preset animation mẫu để LLM tham chiếu.
- [ ] Cải thiện timing khớp animation với lời đọc theo scene.

### 4.2 Hiệu năng render
- [ ] Song song hóa render scene độc lập.
- [ ] Cache frame animation deterministic.
- [ ] Giảm supersample có điều kiện cho scene tĩnh.

### 4.3 Kiểm thử
- [ ] Test tự động cho `llm.py` (mock provider, kiểm tra fallback + 429).
- [ ] Test render golden-image cho `visual_engine` + `mathviz`.
- [ ] Smoke test pipeline `--dry-run` trong CI.

---

## 5. Ưu Tiên Dài Hạn

- [ ] Đa ngôn ngữ (không chỉ tiếng Việt).
- [ ] Chọn chủ đề thông minh (theo trend / analytics YouTube).
- [ ] A/B test thumbnail & tiêu đề.
- [ ] Dashboard theo dõi trạng thái pipeline & lịch sử upload.
- [ ] Hỗ trợ nhiều định dạng (video dài + Short từ cùng một nguồn).

---

## 6. Nợ Kỹ Thuật & Rủi Ro

| Vấn đề | Ảnh hưởng | Hướng xử lý |
| --- | --- | --- |
| `google.generativeai` deprecated | Cảnh báo, có thể vỡ sau này | Migrate sang `google.genai` |
| Gemini free-tier 5 req/phút | Dễ 429 nếu Opus lỗi | Đã có Groq làm tầng 3 |
| Proxy Opus `justwoker.icu` hay 524 | Chậm/lỗi provider chính | Timeout cấu hình + retry 3 lần |
| moviepy ghim `<2.0` | Không lên bản mới được | Chờ migrate khi cần |
| Không có test tự động | Regression khó phát hiện | Xem mục 4.3 |

---

## 7. Nguyên Tắc Phát Triển

- **Không exec code do LLM sinh ra.** Mọi tính năng liên quan code từ LLM phải dùng
  interpreter khai báo an toàn (xem `animation_bridge.make_safe_fn` + `mathviz/custom_scene`).
- **LLM là điểm lỗi.** Luôn có fallback provider và fail-fast hợp lý.
- **Phụ đề burn ở đáy.** Overlay text phải tránh vùng ~18% dưới cùng.
- **Chỉ thay đổi những gì được yêu cầu.** Tránh over-engineering.
