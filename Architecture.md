# System Architecture — Auto-Create-Video-AI-WorkFlow

Hệ thống tự động hóa sản xuất video giáo dục công nghệ & thuật toán trực quan chuẩn chất lượng cao (phong cách 3Blue1Brown, Kurzgesagt, NeetCode), tự động đăng tải lên YouTube.

---

## 1. Tổng Quan Hệ Thống

```
                        ┌──────────────────────────────┐
                        │       Series Manager         │
                        │  (5-Episode Curriculum/Plan) │
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │      Topic Selector          │
                        │ (Domain Rotation & DB Dedup) │
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │    Compact Fact Pack        │
                        │ (Offline match by default)  │
                        │  Optional full web research │
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │        Script Writer         │
                        │  (LLM Prompting & Micro-trace│
                        │   Part 1 + Part 2 Auto-split)│
                        └──────────────┬───────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                              ▼
 ┌──────────────┐             ┌──────────────────┐             ┌──────────────┐
 │    TTS       │             │ Visual/Animation │             │ Image/Video  │
 │ (VieNeu-TTS  │             │ 1. MotionGraphics│             │ 1. Pexels    │
 │  / Edge-TTS) │             │ 2. Manim 3B1B    │             │ 2. FLUX AI   │
 │              │             │ 3. Matplotlib    │             │ 3. AI Motion │
 │              │             │ 4. StykAnimate IK│             │              │
 └──────┬───────┘             └────────┬─────────┘             └──────┬───────┘
        │                              │                              │
        └──────────────────────────────┼──────────────────────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │         Compositor           │
                        │ (MoviePy + FFmpeg Mastering) │
                        │   Ken Burns + Audio Crossfade│
                        └──────────────┬───────────────┘
                                       │
                        ┌──────────────┴───────────────┐
                        ▼                              ▼
         ┌──────────────────────────────┐ ┌──────────────────────────┐
         │     Bilingual Subtitles      │ │     Thumbnail AI/SVG     │
         │   (VI + EN .srt generation)  │ │ (Split Layout 40% / 60%) │
         └──────────────┬───────────────┘ └────────────┬─────────────┘
                        │                              │
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │       YouTube Uploader       │
                        │ (YouTube Data API v3 Upload) │
                        └──────────────────────────────┘
```

---

## 2. Các Thành Phần Cốt Lõi (Core Components)

| Thành phần | File chính | Trách nhiệm | Công nghệ |
|---|---|---|---|
| **Series & Topic** | `series_manager.py`, `topic_selector.py` | Lập giáo trình chuỗi 5 tập, luân chuyển 9 domain, chống trùng lặp | SQLite, Neon DB, Python |
| **Fact Checking** | `research_engine.py`, `script_quality.py` | Fact pack offline gọn, kiểm tra phép tính, nguồn tham khảo | Markdown/PDF, deterministic arithmetic |
| **Script Writer** | `script_writer.py` | Tạo kịch bản JSON chia 2 phần (Long/Short), phân cảnh chi tiết | LLM Prompting, JSON-Repair |
| **LLM Router** | `llm.py` | Điều phối multi-provider fallback theo tác vụ (Script/Code/Topic) | Gemini 3.8 Flash, Groq, Cerebras, Pollinations |
| **TTS Engine** | `tts.py` | Chuyển văn bản thành giọng đọc truyền cảm, nhịp điệu chậm rãi | VieNeu-TTS v3 Turbo, Edge-TTS |
| **Motion Graphics** | `motion_graphics.py` | Render đồ họa động HTML5/Canvas, công thức toán MathJax | Edge/Chrome Headless, Canvas API |
| **AI Code Runner** | `ai_code_runner.py` | Sinh và tự sửa mã animation toán học trong subprocess có timeout | Manim Community, Matplotlib, FFmpeg |
| **Video Quality Gate** | `video_quality.py`, `script_quality.py` | Kiểm tra kịch bản trước render, media/subtitle sau render; xuất báo cáo và storyboard | ffprobe, FFmpeg, Pillow |
| **Stick Animation** | `stick_animate.py` | Hoạt họa người que diễn hoạt và linh vật hỗ trợ bài giảng | 2D IK (Law of Cosines, FABRIK), Mascot |
| **Visual Engine** | `visual_engine.py` | Vẽ slide bài giảng, code card, challenge card | Pillow, Matplotlib |
| **Media Fetcher** | `image_fetcher.py`, `ai_video.py` | Tải video footage B-roll và sinh ảnh minh họa nền | Pexels API, FLUX Schnell, Pollinations |
| **Compositor** | `compositor.py` | Ghép audio/visual, zoom Ken Burns, hòa âm BGM/SFX, burn phụ đề | MoviePy, FFmpeg |
| **Subtitles** | `subtitles.py` | Tự động tạo phụ đề song ngữ tiếng Việt & tiếng Anh | SRT Timings, LLM Translation |
| **Thumbnail** | `thumbnail_ai.py`, `thumbnail_svg.py` | Tạo thumbnail chuẩn tỷ lệ vàng chia cột (Split 40/60) | FLUX AI, Edge Headless SVG, Pillow |
| **Uploader** | `youtube_uploader.py` | Tự động đăng video, đặt lịch, gắn tags SEO và phụ đề | Google YouTube Data API v3 |
| **Database** | `db.py` | Quản lý trạng thái video, tập phim chuỗi, và lịch sử upload | SQLite (cục bộ) / Neon PostgreSQL |

---

## 3. Luồng Dữ Liệu & Quy Trình 3 Tầng Fallback

### 3.1 Fact Pack Mặc Định & Nghiên Cứu Web Tùy Chọn
Mặc định, script writer chỉ tìm một trích đoạn ngắn khớp chủ đề trong tài liệu offline; không gọi web. Đặt `research.enabled: true` để bật chuỗi nghiên cứu web, link dự phòng và tài liệu offline bên dưới.

1. **Tầng 1 (Tự động tìm kiếm mạng)**: Tra cứu Wikipedia REST API (VI/EN) $\to$ DuckDuckGo API $\to$ ArXiv API $\to$ TinyFish Agent. Nếu độ tương đồng cao và có thông tin kỹ thuật, lưu cache và nạp vào prompt.
2. **Tầng 2 (Link backup uy tín)**: Nếu Tầng 1 lỗi mạng hoặc không có kết quả, quét `assets/docs/sources.yaml` theo từ khóa domain và tải trực tiếp từ URL chuẩn (CP-Algorithms, Wikipedia Direct, RFCs).
3. **Tầng 3 (Kho tài liệu tải sẵn offline)**: Khi các tầng trước không có kết quả, có thể trích xuất dữ liệu từ `assets/docs/*.md` (hoặc `.pdf`); nội dung vẫn cần được kiểm tra theo chủ đề và không được xem là bảo đảm chính xác tuyệt đối.

### 3.2 Luồng LLM Fallback (Task Routing)
- **Tác vụ viết kịch bản**: `Gemini 3.8 Flash` $\to$ `Gemini 3.7 Flash` $\to$ `Groq Fast` $\to$ `Cerebras` $\to$ các provider dự phòng.
- **Tác vụ viết code Animation**: `Gemini 3.8/3.7 Flash` $\to$ `Cerebras` $\to$ `Groq LPU` $\to$ `Pollinations Coder`.
- **Tác vụ chọn chủ đề**: `Groq Fast` $\to$ `Groq` $\to$ `Gemini` $\to$ `GitHub Models`.

### 3.3 Luồng Hoạt Họa (Animation Fallback)
1. **Motion Graphics (HTML5/Canvas)**: Dành cho sơ đồ hệ thống, cây nhị phân, bảng dữ liệu, công thức MathJax.
2. **Manim (3Blue1Brown)**: Dành cho biến đổi không gian, đại số tuyến tính, hình học trực quan, giải tích.
3. **Matplotlib FuncAnimation**: Dành cho mô phỏng hàm số, phân bố xác suất, đồ thị đường cong neon.
4. **MathViz (Procedural Engine)**: Fallback cuối cùng nội tại đảm bảo video không bao giờ thiếu chuyển động.

---

## 4. Cơ Sở Dữ Liệu & Lưu Trữ (Data Storage)

### Bảng dữ liệu SQLite / Postgres (`db.py`):
- `videos`: thêm `run_key` duy nhất và các trạng thái `content_approved`, `uploaded_private`, `quarantined`, `uploaded`.
- `series`: `id`, `name`, `mode` (`short`, `long`), `total_episodes`, `status` (`in_progress`, `completed`), `created_at`.
- `episodes`: `id`, `series_id`, `episode_num`, `topic`, `video_id`, `status`.
- `pipeline_runs`, `stage_runs`: checkpoint và idempotency cho từng stage production.

---

## 5. An Toàn Hệ Thống & Nguyên Tắc Bất Biến

1. **Kiểm tra mã số học**: Các biểu thức modular arithmetic có dạng `a^b mod n = r` được tính lại trước render; các khẳng định ngoài phạm vi kiểm tra vẫn cần biên tập soát lại.
2. **AI animation runner**: Mã Manim qua AST policy rồi chạy trong container tạm
   không mạng/secret, rootfs read-only, non-root và có resource limits. Code vi phạm
   policy bị từ chối; Matplotlib-AI tắt mặc định.
3. **Safe Margins**: Vùng văn bản và phụ đề luôn cách đáy tối thiểu 18% để tránh bị che khuất bởi giao diện YouTube Shorts / Player Controls.
4. **Publish gate**: Video luôn staging private và chỉ tự public sau khi content QA,
   scene QA, final QA và YouTube processing đều pass.
