# Production pipeline

Luồng production không cần người duyệt thường xuyên:

```text
topic → script → content QA → visual plan → TTS/render → scene QA
      → compose → final QA → upload private → YouTube processing
      → machine publish policy → public hoặc quarantine
```

## Bất biến production

- `APP_ENV=production`.
- `DATABASE_URL` phải là PostgreSQL; lỗi DB làm run dừng, không fallback SQLite.
- `animation.sandbox.backend=container`.
- Code Manim phải qua `manim_validator` và chạy trong image
  `docker/manim-worker/Dockerfile` với network tắt.
- YouTube staging luôn `private`.
- `run_key` chống tạo/upload trùng; run bị ngắt sau upload sẽ tiếp tục publish video
  cũ thay vì upload bản mới.

## Báo cáo của mỗi run

- `content_quality_report.json`: accuracy, clarity, logic, pedagogy, Vietnamese và retention.
- `storyboard_plan.json`: SceneContract do Visual Director tạo.
- `scene_quality_report.json`: asset, text budget và animation của từng scene.
- `quality_report.json`: stream, duration, resolution, FPS, subtitle, black/freeze scan.
- `run_state.json`: checkpoint local; PostgreSQL giữ bản authoritative.
- `manifest.json`: kết quả upload, processing và machine publish decision.

## Khi video bị quarantine

Không public thủ công trước khi đọc các báo cáo trên. Sửa nguyên nhân hoặc khôi phục
dịch vụ phụ thuộc rồi chạy lại cùng ngày/episode; idempotency sẽ tiếp tục đúng video
private đã upload nếu có.
