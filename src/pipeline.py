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
from .compositor import compose
from .config import CONFIG, OUTPUT_DIR
from .metadata import build_metadata, make_thumbnail
from .models import Script
from .script_writer import write_script
from .subtitles import generate_srt
from .topic_selector import pick_topic
from .tts import synthesize

log = logging.getLogger(__name__)


def _render_video(script: Script, workdir: Path) -> tuple[Path, Path]:
    """Render kịch bản thành file video + thumbnail. Trả về (video, thumbnail)."""
    from .visual_engine import render_scene

    images: list[Path] = []
    audios: list[Path] = []
    full_narration: list[str] = []

    for i, scene in enumerate(script.scenes):
        img = workdir / f"scene_{i:02d}.png"
        aud = workdir / f"scene_{i:02d}.wav"
        render_scene(scene, img)
        synthesize(scene.narration, aud)
        images.append(img)
        audios.append(aud)
        full_narration.append(scene.narration)

    # phụ đề: ghép audio đã có sẵn timing theo từng scene -> whisper toàn bộ
    srt_path: Path | None = None
    if CONFIG["subtitles"].get("enabled"):
        try:
            # dùng audio scene đầu tiên là không đủ; whisper chạy trên video sau ghép
            merged_audio = workdir / "narration_full.wav"
            _concat_wav(audios, merged_audio)
            srt_path = generate_srt(merged_audio, workdir / "subs.srt")
        except Exception as e:  # noqa: BLE001 - phụ đề không bắt buộc
            log.warning("Sinh phụ đề lỗi: %s", e)
            srt_path = None

    video_path = compose(images, audios, workdir / "video.mp4", srt_path)

    thumb_path = make_thumbnail(script, workdir / "thumbnail.png")
    return video_path, thumb_path


def _concat_wav(wavs: list[Path], out: Path) -> None:
    import wave

    with wave.open(str(wavs[0]), "rb") as w0:
        params = w0.getparams()
    with wave.open(str(out), "wb") as wout:
        wout.setparams(params)
        for p in wavs:
            with wave.open(str(p), "rb") as w:
                wout.writeframes(w.readframes(w.getnframes()))


def run_once(upload_video: bool = True, dry_run: bool = False) -> None:
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
    args = parser.parse_args()

    n = int(CONFIG.get("videos_per_run", 1))
    for i in range(n):
        log.info("=== Video %d/%d ===", i + 1, n)
        run_once(upload_video=not args.no_upload, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
