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
                        │       Research Engine        │
                        │  (3-Tier Fallback Knowledge) │
                        │  1. Web/Wiki/ArXiv/DDG       │
                        │  2. Curated sources.yaml     │
                        │  3. assets/docs/ Offline DB  │
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
| **Research Engine** | `research_engine.py` | Tra cứu tài liệu 3 tầng fallback, cung cấp công thức $0$ ảo giác | Wikipedia REST API, DuckDuckGo, ArXiv, Markdown/PDF |
| **Script Writer** | `script_writer.py` | Tạo kịch bản JSON chia 2 phần (Long/Short), phân cảnh chi tiết | LLM Prompting, JSON-Repair |
| **LLM Router** | `llm.py` | Điều phối multi-provider fallback theo tác vụ (Script/Code/Topic) | Opus, Gemini 3.7, Groq, Cerebras, Pollinations |
| **TTS Engine** | `tts.py` | Chuyển văn bản thành giọng đọc truyền cảm, nhịp điệu chậm rãi | VieNeu-TTS v3 Turbo, Edge-TTS |
| **Motion Graphics** | `motion_graphics.py` | Render đồ họa động HTML5/Canvas, công thức toán MathJax | Edge/Chrome Headless, Canvas API |
| **AI Code Runner** | `ai_code_runner.py` | Sinh và tự sửa mã animation toán học (sandbox an toàn) | Manim Community, Matplotlib, FFmpeg |
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

### 3.1 Luồng Nghiên Cứu Tài Liệu (Research Fallback)
1. **Tầng 1 (Tự động tìm kiếm mạng)**: Tra cứu Wikipedia REST API (VI/EN) $\to$ DuckDuckGo API $\to$ ArXiv API $\to$ TinyFish Agent. Nếu độ tương đồng cao và có thông tin kỹ thuật, lưu cache và nạp vào prompt.
2. **Tầng 2 (Link backup uy tín)**: Nếu Tầng 1 lỗi mạng hoặc không có kết quả, quét `assets/docs/sources.yaml` theo từ khóa domain và tải trực tiếp từ URL chuẩn (CP-Algorithms, Wikipedia Direct, RFCs).
3. **Tầng 3 (Kho tài liệu tải sẵn offline)**: Nếu không có Internet hoặc cả 2 tầng trên thất bại, trích xuất tức thì từ `assets/docs/*.md` (hoặc `.pdf`) với công thức toán và bộ số liệu đã kiểm chứng 100%.

### 3.2 Luồng LLM Fallback (Task Routing)
- **Tác vụ viết kịch bản**: `Claude Opus` $\to$ `Gemini 3.7 Flash` $\to$ `Groq Llama 3.3` $\to$ `Cerebras` $\to$ `Pollinations`.
- **Tác vụ viết code Animation**: `Claude Opus` $\to$ `Cerebras` $\to$ `Groq LPU` $\to$ `Gemini 3.7` $\to$ `Pollinations Coder`.
- **Tác vụ chọn chủ đề**: `Gemini 3.7 Flash` $\to$ `Groq Fast` $\to$ `GitHub Models`.

### 3.3 Luồng Hoạt Họa (Animation Fallback)
1. **Motion Graphics (HTML5/Canvas)**: Dành cho sơ đồ hệ thống, cây nhị phân, bảng dữ liệu, công thức MathJax.
2. **Manim (3Blue1Brown)**: Dành cho biến đổi không gian, đại số tuyến tính, hình học trực quan, giải tích.
3. **Matplotlib FuncAnimation**: Dành cho mô phỏng hàm số, phân bố xác suất, đồ thị đường cong neon.
4. **MathViz (Procedural Engine)**: Fallback cuối cùng nội tại đảm bảo video không bao giờ thiếu chuyển động.

---

## 4. Cơ Sở Dữ Liệu & Lưu Trữ (Data Storage)

### Bảng dữ liệu SQLite / Postgres (`db.py`):
- `videos`: `id`, `topic`, `title`, `status` (`created`, `scripted`, `rendered`, `uploaded`), `youtube_id`, `created_at`.
- `series`: `id`, `name`, `mode` (`short`, `long`), `total_episodes`, `status` (`in_progress`, `completed`), `created_at`.
- `episodes`: `id`, `series_id`, `episode_num`, `topic`, `video_id`, `status`.

---

## 5. An Toàn Hệ Thống & Nguyên Tắc Bất Biến

1. **Sandboxing cho AI Code**: Mọi mã Python sinh ra bởi LLM (Manim, Matplotlib) chỉ được chạy trong subprocess cô lập có giới hạn timeout cứng, tuyệt đối không chạy `eval()` / `exec()` trong tiến trình chính.
2. **Zero Hallucination**: Toàn bộ công thức toán học, modulo, số mũ phải tuân theo dữ liệu chuẩn xác của Research Engine.
3. **Safe Margins**: Vùng văn bản và phụ đề luôn cách đáy tối thiểu 18% để tránh bị che khuất bởi giao diện YouTube Shorts / Player Controls.
