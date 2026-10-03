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


def _exercise_scenes(script: Script) -> list:
    """Tạo DUY NHẤT một scene "thử thách" từ bài toán ĐẦU TIÊN (nếu LLM sinh nhiều).

    Vị trí chèn do ``_with_exercise`` quyết định (trước phần tổng kết).
    """
    from .models import Scene

    exercises = getattr(script, "exercises", None) or []
    if not exercises:
        return []

    ex = exercises[0]
    bullets = [ex.question]
    if ex.hint:
        bullets.append("Gợi ý: " + ex.hint)
    narration = f"Thử thách nhỏ cho bạn: {ex.question}"
    if ex.hint:
        narration += f" Gợi ý: {ex.hint}"
    narration += " Hãy để lại đáp án của bạn ở phần bình luận."
    return [
        Scene(
            narration=narration,
            visual_type="challenge",
            heading="Thử thách",
            bullets=bullets,
            image_query="real world math application",
        )
    ]


def _with_exercise(script: Script) -> list:
    """Danh sách scene render: chèn scene thử thách TRƯỚC 2 scene cuối (tổng kết + mở
    sang video sau) thay vì sau call-to-action — video "đã kết thúc" rồi lại hiện slide
    chữ tĩnh khiến người xem thoát ngay. Short bỏ qua để không vượt thời lượng mục tiêu.
    """
    scenes = list(script.scenes)
    if CONFIG.get("active_mode") == "short":
        return scenes
    extra = _exercise_scenes(script)
    if not extra:
        return scenes
    pos = len(scenes) - 2 if len(scenes) >= 3 else len(scenes)
    return scenes[:pos] + extra + scenes[pos:]


def _render_video(script: Script, workdir: Path) -> tuple[Path, Path, list[float]]:
    """Render kịch bản thành file video + thumbnail. Trả về (video, thumbnail)."""
    # Import trễ để apply_mode (đổi W/H) có hiệu lực trước khi module cache kích thước
    from .visual_engine import render_scene, render_overlay
    from .animation_bridge import build_animation_scene
    from .compositor import compose
    from .metadata import make_thumbnail
    from .subtitles import srt_from_scenes
    from .image_fetcher import fetch_video

    import wave

    is_short = CONFIG.get("active_mode") == "short"
    video_cfg = CONFIG.get("videos", {}) or {}
    broll_enabled = bool(video_cfg.get("enabled", False))
    broll_max = int(video_cfg.get("max_per_video", 6))
    broll_used = 0
    images: list[Path] = []
    audios: list[Path] = []
    scene_texts: list[str] = []
    durations: list[float] = []
    anim_scenes: list = []
    broll_videos: list = []
    scene_overlays: list = []
    code_videos: list = []

    from .tts import synthesize_timed
    from .models import Scene

    # Chọn lại provider TTS từ đầu cho video này rồi khóa -> cả video 1 giọng.
    from .tts import reset_provider_lock
    reset_provider_lock()

    # Scene chính + scene "thử thách" chèn trước phần tổng kết.
    render_scenes = _with_exercise(script)
    scene_timings: list = []

    for i, scene in enumerate(render_scenes):
        img = workdir / f"scene_{i:02d}.png"
        aud = workdir / f"scene_{i:02d}.wav"
        # Luôn render ảnh tĩnh làm fallback
        render_scene(scene, img)
        # Tổng hợp audio TRƯỚC để lấy đúng thời lượng cho animation.
        # Đọc từng câu -> biết mốc thật của mỗi câu cho phụ đề.
        scene_timings.append(synthesize_timed(scene.narration, aud))
        with wave.open(str(aud), "rb") as w:
            dur = w.getnframes() / float(w.getframerate())
        durations.append(dur)
        images.append(img)
        audios.append(aud)
        scene_texts.append(scene.narration)

        # Scene động: nếu visual_type == "animation" và có cấu hình
        anim = None
        code_video = None
        acfg = scene.animation or {}
        if scene.visual_type == "animation":
            from .ai_code_runner import run_ai_code, run_manim_code, generate_matplotlib_from_prompt

            preset = str(acfg.get("preset", ""))
            anim_cfg = CONFIG.get("animation", {}) or {}

            # Tầng 1: Manim nếu preset yêu cầu và được bật
            if preset == "manim" and acfg.get("code") and anim_cfg.get("manim_enabled", True):
                try:
                    code_video = run_manim_code(
                        str(acfg["code"]),
                        workdir / f"aimanim_{i:02d}",
                        duration=dur,
                        width=CONFIG["visual"]["width"],
                        height=CONFIG["visual"]["height"],
                        fps=int(anim_cfg.get("manim_fps", 60)),
                        quality=str(anim_cfg.get("manim_quality", "high_quality")),
                        background_color=str(CONFIG["visual"].get("background_color", "#0d1117")),
                        timeout=int(anim_cfg.get("manim_timeout", 300)),
                        auto_repair=True,
                    )
                except Exception as e:  # noqa: BLE001
                    log.warning("Manim scene %d lỗi: %s", i, e)

            # Tầng 2: PyCode (Matplotlib) nếu đã có code hoặc Manim thất bại
            if code_video is None:
                pycode = acfg.get("code") if preset == "pycode" else acfg.get("pycode")
                # Nếu preset là manim/pycode nhưng chưa có code -> Nhờ AI sinh code Matplotlib
                if not pycode and (preset in ("manim", "pycode") or scene.narration):
                    desc = scene.heading or scene.narration[:120]
                    pycode = generate_matplotlib_from_prompt(desc, duration=dur)

                if pycode:
                    try:
                        code_video = run_ai_code(
                            str(pycode),
                            workdir / f"aicode_{i:02d}",
                            duration=dur,
                            width=CONFIG["visual"]["width"],
                            height=CONFIG["visual"]["height"],
                            fps=CONFIG["visual"]["fps"],
                            auto_repair=True,
                        )
                    except Exception as e:  # noqa: BLE001
                        log.warning("Matplotlib scene %d lỗi: %s", i, e)

            # Tầng 3 (CUỐI CÙNG): MathViz Preset / Custom objects
            if code_video is None and acfg:
                try:
                    anim = build_animation_scene(scene, dur)
                except Exception as e:  # noqa: BLE001
                    log.warning("MathViz fallback scene %d lỗi: %s", i, e)
                    anim = None
        anim_scenes.append(anim)
        code_videos.append(code_video)

        # Scene b-roll: tải video footage minh họa (chỉ khi không phải animation
        # và scene có video_query). Nếu không có Pexels key/không tải được -> None.
        broll = None
        overlay = None
        if broll_enabled and broll_used < broll_max and anim is None and code_video is None and scene.video_query:
            try:
                broll = fetch_video(scene.video_query, index=i % 3, vertical=is_short)
            except Exception as e:  # noqa: BLE001
                log.debug("Tải b-roll scene %d lỗi: %s", i, e)
                broll = None
            if broll is not None:
                broll_used += 1
                overlay = workdir / f"overlay_{i:02d}.png"
                try:
                    render_overlay(scene, overlay)
                except Exception as e:  # noqa: BLE001
                    log.warning("Render overlay scene %d lỗi: %s", i, e)
                    overlay = None
        broll_videos.append(broll)
        scene_overlays.append(overlay)

    # Phụ đề: dùng chính text narration gốc (chính xác 100%), căn theo thời lượng scene
    srt_path: Path | None = None
    captions_dict: dict[str, Path] = {}
    if CONFIG["subtitles"].get("enabled"):
        try:
            srt_path = srt_from_scenes(scene_texts, durations, workdir / "subs_vi.srt", scene_timings)
            captions_dict["vi"] = srt_path
            from .subtitles import translate_srt_to_english

            srt_en = translate_srt_to_english(srt_path, workdir / "subs_en.srt")
            if srt_en and srt_en.exists():
                captions_dict["en"] = srt_en
        except Exception as e:  # noqa: BLE001 - phụ đề không bắt buộc
            log.warning("Sinh phụ đề lỗi: %s", e)
            srt_path = None

    video_path = compose(
        images, audios, workdir / "video.mp4", srt_path,
        anim_scenes, broll_videos, scene_overlays, code_videos,
    )

    thumb_path = workdir / "thumbnail.png"
    made = None
    # 1. Ưu tiên Thumbnail SVG công nghệ chuẩn theo thiết kế tối giản mới
    try:
        from .thumbnail_svg import render_svg_thumbnail
        made = render_svg_thumbnail(script, thumb_path)
    except Exception as e:  # noqa: BLE001
        log.warning("Thumbnail SVG lỗi (%s) -> thử AI Kurzgesagt", e)

    # 2. Thumbnail AI phong cách Kurzgesagt nếu SVG fail và bật ai_enabled
    if made is None and CONFIG.get("thumbnail", {}).get("ai_enabled"):
        try:
            from .thumbnail_ai import make_ai_thumbnail
            made = make_ai_thumbnail(script, thumb_path)
        except Exception as e:  # noqa: BLE001
            log.warning("Thumbnail AI thất bại (%s) -> dùng thumbnail thường", e)

    # 3. Fallback cuối cùng: make_thumbnail
    if made is None:
        thumb_path = make_thumbnail(script, thumb_path)
    else:
        thumb_path = made
    return video_path, thumb_path, durations, captions_dict


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

        video_path, thumb_path, durations, captions = _render_video(script, workdir)
        db.update_video(video_id_db, status="rendered")

        if not upload_video:
            log.info("Đã render (không upload): %s", video_path)
            db.update_video(video_id_db, status="rendered_local")
            return

        from .youtube_uploader import upload
        from .metadata import build_metadata

        meta = build_metadata(script, durations)
        yt_id = upload(video_path, meta, thumb_path, captions=captions)
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
