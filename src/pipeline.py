"""Orchestrator: chạy toàn bộ pipeline tạo + upload video.

Chạy: python -m src.pipeline            (tạo & upload theo config)
      python -m src.pipeline --no-upload (chỉ render, không upload - để test)
      python -m src.pipeline --dry-run   (chỉ in kịch bản, không render)
"""
from __future__ import annotations

import argparse
import logging
import traceback
from pathlib import Path

from . import db
from .config import CONFIG, OUTPUT_DIR, apply_mode
from .models import Script

log = logging.getLogger(__name__)


def _render_video(script: Script, workdir: Path) -> tuple[Path, Path]:
    """Render kịch bản thành file video + thumbnail. Trả về (video, thumbnail)."""
    # Import trễ để apply_mode (đổi W/H) có hiệu lực trước khi module cache kích thước
    from .visual_engine import render_scene
    from .animation_bridge import build_animation_scene
    from .compositor import compose
    from .metadata import make_thumbnail
    from .subtitles import srt_from_scenes

    import wave

    images: list[Path] = []
    audios: list[Path] = []
    scene_texts: list[str] = []
    durations: list[float] = []
    anim_scenes: list = []

    from .tts import synthesize

    for i, scene in enumerate(script.scenes):
        img = workdir / f"scene_{i:02d}.png"
        aud = workdir / f"scene_{i:02d}.wav"
        # Luôn render ảnh tĩnh làm fallback
        render_scene(scene, img)
        # Tổng hợp audio TRƯỚC để lấy đúng thời lượng cho animation
        synthesize(scene.narration, aud)
        with wave.open(str(aud), "rb") as w:
            dur = w.getnframes() / float(w.getframerate())
        durations.append(dur)
        images.append(img)
        audios.append(aud)
        scene_texts.append(scene.narration)

        # Scene động: nếu visual_type == "animation" và có cấu hình
        anim = None
        if scene.visual_type == "animation" and scene.animation:
            try:
                anim = build_animation_scene(scene, dur)
            except Exception as e:  # noqa: BLE001
                log.warning("Bỏ animation scene %d: %s", i, e)
                anim = None
        anim_scenes.append(anim)

    # Phụ đề: dùng chính text narration gốc (chính xác 100%), căn theo thời lượng scene
    srt_path: Path | None = None
    if CONFIG["subtitles"].get("enabled"):
        try:
            srt_path = srt_from_scenes(scene_texts, durations, workdir / "subs.srt")
        except Exception as e:  # noqa: BLE001 - phụ đề không bắt buộc
            log.warning("Sinh phụ đề lỗi: %s", e)
            srt_path = None

    video_path = compose(images, audios, workdir / "video.mp4", srt_path, anim_scenes)

    thumb_path = make_thumbnail(script, workdir / "thumbnail.png")
    return video_path, thumb_path


def run_once(upload_video: bool = True, dry_run: bool = False) -> None:
    from .script_writer import write_script
    from .topic_selector import pick_topic

    db.init_db()

    topic = pick_topic()
    video_id_db = db.create_video(topic)

    try:
        script = write_script(topic)
        db.update_video(video_id_db, title=script.title, status="scripted")

        if dry_run:
            print(script.model_dump_json(indent=2))
            db.update_video(video_id_db, status="dry_run")
            return

        workdir = OUTPUT_DIR / f"video_{video_id_db}"
        workdir.mkdir(parents=True, exist_ok=True)

        video_path, thumb_path = _render_video(script, workdir)
        db.update_video(video_id_db, status="rendered")

        if not upload_video:
            log.info("Đã render (không upload): %s", video_path)
            db.update_video(video_id_db, status="rendered_local")
            return

        from .youtube_uploader import upload
        from .metadata import build_metadata

        meta = build_metadata(script)
        yt_id = upload(video_path, meta, thumb_path)
        db.update_video(video_id_db, status="uploaded", youtube_id=yt_id)
        log.info("HOÀN TẤT: https://youtu.be/%s", yt_id)

    except Exception as e:  # noqa: BLE001 - ghi lỗi vào DB rồi raise
        db.update_video(video_id_db, status="error", error=str(e)[:500])
        log.error("Pipeline lỗi: %s\n%s", e, traceback.format_exc())
        raise


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-upload", action="store_true", help="Chỉ render, không upload")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ in kịch bản")
    parser.add_argument(
        "--mode",
        choices=["long", "short"],
        default=None,
        help="long = video dài ngang 16:9 (buổi sáng), short = dọc 9:16 <60s (buổi tối)",
    )
    args = parser.parse_args()

    mode = apply_mode(args.mode)
    log.info(
        "Mode: %s (%dx%d, %ds)",
        mode,
        CONFIG["visual"]["width"],
        CONFIG["visual"]["height"],
        CONFIG["target_duration_seconds"],
    )

    n = int(CONFIG.get("videos_per_run", 1))
    for i in range(n):
        log.info("=== Video %d/%d ===", i + 1, n)
        run_once(upload_video=not args.no_upload, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
