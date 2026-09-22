# Tech Viz Bot — Tự tạo & upload video tech mỗi ngày

App tự chọn chủ đề công nghệ, viết kịch bản, render video visualization bằng code
(Matplotlib + Pillow), lồng tiếng, gắn phụ đề, rồi tự upload lên YouTube.
Chạy tự động 2 lần/ngày bằng GitHub Actions — không tốn VPS, không cần ra lệnh.

## Kiến trúc

```
Chọn chủ đề (AI) → Viết kịch bản (LLM) → Render scene (Pillow/Matplotlib)
   → Lồng tiếng (TTS) → Phụ đề (Whisper) → Ghép video (ffmpeg)
   → Metadata + thumbnail → Upload YouTube
```

| Thành phần | Công nghệ |
|---|---|
| LLM | Claude Opus (chính) → Gemini Flash (fallback) |
| TTS | VieNeu-TTS → ElevenLabs → Edge-TTS (fallback tự động) |
| Visual | Pillow + Matplotlib |
| Phụ đề | faster-whisper |
| Render | moviepy + ffmpeg |
| Upload | YouTube Data API v3 |
| Deploy | GitHub Actions (cron 2×/ngày) |

## Cài đặt local

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # điền API key
```

Cần cài `ffmpeg` trên máy (Windows: `choco install ffmpeg` hoặc tải từ ffmpeg.org).

## Lấy API key

- **Gemini**: https://aistudio.google.com/apikey → điền `GEMINI_API_KEY`
- **Claude Opus**: https://console.anthropic.com → điền `ANTHROPIC_API_KEY`
- **ElevenLabs** (tùy chọn): https://elevenlabs.io → `ELEVENLABS_API_KEY`

## Thiết lập YouTube OAuth (làm 1 lần)

1. Vào https://console.cloud.google.com → tạo project.
2. Bật **YouTube Data API v3**.
3. Tạo **OAuth client ID** loại *Desktop app*, tải về, lưu thành
   `credentials/client_secret.json`.
4. Chạy để xác thực và tạo token:
   ```bash
   python -m src.youtube_uploader --auth
   ```
   Trình duyệt mở ra, đăng nhập kênh YouTube muốn đăng. Lệnh in ra chuỗi token JSON.

## Test local (không upload)

```bash
python -m src.pipeline --dry-run    # chỉ in kịch bản
python -m src.pipeline --no-upload  # render ra output/, không đăng
```

Video nằm ở `output/video_<id>/video.mp4`.

## Deploy lên GitHub Actions

1. Push code lên GitHub repo.
2. Vào **Settings → Secrets and variables → Actions**, thêm các secret:

   | Secret | Giá trị |
   |---|---|
   | `GEMINI_API_KEY` | key Gemini |
   | `ANTHROPIC_API_KEY` | key Opus |
   | `ELEVENLABS_API_KEY` | key ElevenLabs (tùy chọn) |
   | `YOUTUBE_CLIENT_SECRET` | nội dung `client_secret.json` (dán nguyên JSON) |
   | `YOUTUBE_TOKEN` | chuỗi token in ra từ lệnh `--auth` |

3. Workflow tự chạy lúc **00:00 và 12:00 UTC** (7h & 19h giờ VN).
   Chạy tay: tab **Actions → Tạo & upload video tech → Run workflow**.

## Tùy chỉnh

Sửa `config.yaml`:
- `videos_per_run`: số video mỗi lần chạy.
- `language`: `vi` hoặc `en`.
- `target_duration_seconds`: độ dài video.
- `topics.domains`: danh sách lĩnh vực để AI chọn chủ đề.
- `youtube.privacy_status`: `public` / `unlisted` / `private`.

Muốn đúng "2 video/ngày": để `videos_per_run: 1` và giữ 2 mốc cron (mỗi lần 1 video),
hoặc để `videos_per_run: 2` và 1 mốc cron.

## Lưu ý quan trọng

- **Quota YouTube**: mỗi upload ~1.600/10.000 units/ngày → 2 video/ngày thoải mái.
- **Chính sách YouTube**: nội dung tự động cần có giá trị thật, tránh bị gắn cờ spam.
- **Nhạc/font**: bỏ file vào `assets/music` và `assets/fonts` (xem README trong đó).
- **`output/state.db`** được commit lại để nhớ chủ đề đã dùng, chống trùng lặp.
