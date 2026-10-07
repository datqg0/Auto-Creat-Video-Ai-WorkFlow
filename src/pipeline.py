"""Orchestrator: chạy toàn bộ pipeline tạo + upload video.

Chạy: python -m src.pipeline            (tạo & upload theo config)
      python -m src.pipeline --no-upload (chỉ render, không upload - để test)
      python -m src.pipeline --dry-run   (chỉ in kịch bản, không render)
"""
from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

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

    ai_vid_cfg = CONFIG.get("ai_video", {}) or {}
    ai_video_enabled = bool(ai_vid_cfg.get("enabled", True))
    max_ai_videos = int(ai_vid_cfg.get("max_per_run", 1))
    ai_video_used = 0
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

        # Scene động: nếu visual_type == "animation" hoặc bật animate_from_start cho scene mở đầu
        anim = None
        code_video = None
        acfg = scene.animation or {}
        anim_cfg = CONFIG.get("animation", {}) or {}
        is_anim = (scene.visual_type == "animation") or (i == 0 and anim_cfg.get("animate_from_start", True))

        if is_anim:
            from .ai_code_runner import run_ai_code, run_manim_code, generate_matplotlib_from_prompt, generate_manim_from_prompt

            preset = str(acfg.get("preset", ""))
            visual_desc = scene.visual_prompt or scene.heading or scene.narration[:120]

            # Tầng 1: Manim nếu preset yêu cầu và được bật
            if anim_cfg.get("manim_enabled", True) and (preset == "manim" or "manim" in visual_desc.lower()):
                manim_code = acfg.get("code") if preset == "manim" else None
                if not manim_code and visual_desc:
                    manim_code = generate_manim_from_prompt(
                        visual_desc,
                        duration=dur,
                        topic=script.title,
                        narration=scene.narration,
                    )
                if manim_code:
                    try:
                        code_video = run_manim_code(
                            str(manim_code),
                            workdir / f"aimanim_{i:02d}",
                            duration=dur,
                            width=CONFIG["visual"]["width"],
                            height=CONFIG["visual"]["height"],
                            fps=int(anim_cfg.get("manim_fps", 30)),
                            quality=str(anim_cfg.get("manim_quality", "medium_quality")),
                            background_color=str(CONFIG["visual"].get("background_color", "#0d1117")),
                            timeout=int(anim_cfg.get("manim_timeout", 300)),
                            auto_repair=True,
                        )
                    except Exception as e:  # noqa: BLE001
                        log.warning("Manim scene %d lỗi: %s", i, e)

            # Tầng 2: PyCode (Matplotlib) - AI sinh code Python animation trực quan
            if code_video is None:
                pycode = acfg.get("code") if preset == "pycode" else acfg.get("pycode")
                # Nếu chưa có code -> Nhờ AI chuyên code sinh Matplotlib từ visual_prompt
                if not pycode and (preset in ("manim", "pycode") or anim_cfg.get("ai_code_primary", True) or scene.visual_prompt or scene.narration):
                    pycode = generate_matplotlib_from_prompt(
                        visual_desc,
                        duration=dur,
                        topic=script.title,
                        narration=scene.narration,
                    )

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

        # Scene b-roll / AI Video: ưu tiên tạo tối đa 1 video AI cho mỗi lần chạy pipeline
        broll = None
        overlay = None

        # 1. Thử sinh Video AI nếu chưa đủ hạn mức (tối đa max_ai_videos = 1 clip/video)
        if ai_video_enabled and ai_video_used < max_ai_videos and anim is None and code_video is None:
            v_prompt = scene.video_query or scene.image_query or scene.heading or scene.narration[:60]
            if v_prompt:
                try:
                    from .ai_video import generate_ai_video

                    ai_vid = generate_ai_video(
                        prompt=v_prompt,
                        out_dir=workdir / f"aivideo_{i:02d}.mp4",
                        duration=dur,
                        vertical=is_short,
                    )
                    if ai_vid and ai_vid.exists():
                        broll = ai_vid
                        ai_video_used += 1
                        log.info("Scene %d: Dùng Video AI gen (%s)", i, ai_vid.name)
                except Exception as e:  # noqa: BLE001
                    log.warning("Tạo AI video scene %d lỗi: %s", i, e)

        # 2. Nếu chưa có b-roll từ AI -> tải footage từ Pexels như cũ
        if broll is None and broll_enabled and broll_used < broll_max and anim is None and code_video is None and scene.video_query:
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
    thumb_cfg = CONFIG.get("thumbnail", {}) or {}
    engine = thumb_cfg.get("engine", "ai")

    if engine == "ai" and thumb_cfg.get("ai_enabled", True):
        # 1. Ưu tiên Thumbnail AI bố cục split (chữ trái 40%, ảnh AI phải 60%)
        try:
            from .thumbnail_ai import make_ai_thumbnail
            made = make_ai_thumbnail(script, thumb_path)
        except Exception as e:  # noqa: BLE001
            log.warning("Thumbnail AI thất bại (%s) -> thử fallback SVG", e)

        # 2. Fallback SVG nếu AI không khả dụng
        if made is None:
            try:
                from .thumbnail_svg import render_svg_thumbnail
                made = render_svg_thumbnail(script, thumb_path)
            except Exception as e:  # noqa: BLE001
                log.warning("Thumbnail SVG lỗi (%s) -> dùng thumbnail thường", e)
    else:
        # Mặc định SVG nếu cấu hình engine == 'svg'
        try:
            from .thumbnail_svg import render_svg_thumbnail
            made = render_svg_thumbnail(script, thumb_path)
        except Exception as e:  # noqa: BLE001
            log.warning("Thumbnail SVG lỗi (%s) -> thử AI", e)

        if made is None and thumb_cfg.get("ai_enabled", False):
            try:
                from .thumbnail_ai import make_ai_thumbnail
                made = make_ai_thumbnail(script, thumb_path)
            except Exception as e:  # noqa: BLE001
                log.warning("Thumbnail AI thất bại (%s) -> dùng thumbnail thường", e)

    # 3. Fallback cuối cùng: make_thumbnail (PIL cơ bản)
    if made is None:
        thumb_path = make_thumbnail(script, thumb_path)
    else:
        thumb_path = made
    return video_path, thumb_path, durations, captions_dict


def run_once(
    topic: str | None = None,
    upload_video: bool = True,
    dry_run: bool = False,
) -> None:
    from .script_writer import write_script
    from .series_manager import get_or_create_active_series_task

    db.init_db()

    series_context = None
    episode_id = None
    series_info = None

    if not topic:
        # Tự động điều phối theo Chuỗi Video (Series):
        # Nếu DB trống hoặc series cũ xong -> Tự động sinh series mới!
        try:
            series_task = get_or_create_active_series_task()
            topic = series_task["episode"]["topic"]
            series_context = series_task["series_context"]
            episode_id = series_task["episode"]["id"]
            series_info = series_task["series"]
            db.update_episode(episode_id, status="in_progress")
            log.info(
                "▶ Bắt đầu sản xuất: %s - [Tập %d/%d] %s",
                series_context["series_name"],
                series_context["episode_num"],
                series_context["total_episodes"],
                topic,
            )
        except Exception as e:
            log.warning("Không thể lấy nhiệm vụ series (%s), fallback sang pick_topic đơn lẻ", e)
            from .topic_selector import pick_topic

            topic = pick_topic()

    video_id_db = db.create_video(topic)
    if episode_id:
        db.update_episode(episode_id, video_id=video_id_db)

    try:
        script = write_script(topic, series_context=series_context)
        db.update_video(video_id_db, title=script.title, status="scripted")
        if episode_id:
            db.update_episode(episode_id, status="scripted")

        if dry_run:
            print(script.model_dump_json(indent=2))
            db.update_video(video_id_db, status="dry_run")
            if episode_id:
                db.update_episode(episode_id, status="dry_run")
            return

        workdir = OUTPUT_DIR / f"video_{video_id_db}"
        workdir.mkdir(parents=True, exist_ok=True)

        video_path, thumb_path, durations, captions = _render_video(script, workdir)
        db.update_video(video_id_db, status="rendered")
        if episode_id:
            db.update_episode(episode_id, status="rendered")

        if not upload_video:
            log.info("Đã render (không upload): %s", video_path)
            db.update_video(video_id_db, status="rendered_local")
            if episode_id and series_context and series_info:
                if series_context["episode_num"] >= series_context["total_episodes"]:
                    db.update_series(series_info["id"], status="completed")
                    log.info("🎉 CHÚC MỪNG: Chuỗi video '%s' đã hoàn thành tất cả các tập!", series_info["name"])
            return

        from .youtube_uploader import upload
        from .metadata import build_metadata

        meta = build_metadata(script, durations)
        yt_id = upload(video_path, meta, thumb_path, captions=captions)
        db.update_video(video_id_db, status="uploaded", youtube_id=yt_id)
        if episode_id:
            db.update_episode(episode_id, status="uploaded")
            if series_context and series_info:
                if series_context["episode_num"] >= series_context["total_episodes"]:
                    db.update_series(series_info["id"], status="completed")
                    log.info("🎉 CHÚC MỪNG: Chuỗi video '%s' đã hoàn tất tải lên tất cả các tập!", series_info["name"])
        log.info("HOÀN TẤT: https://youtu.be/%s", yt_id)

    except Exception as e:  # noqa: BLE001 - ghi lỗi vào DB rồi raise
        db.update_video(video_id_db, status="error", error=str(e)[:500])
        if episode_id:
            db.update_episode(episode_id, status="error")
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
        choices=["long", "short", "mega"],
        default=None,
        help="long = video 16:9 (~5p), short = dọc 9:16 (tối đa 3p / 180s), mega = siêu dài 16:9 (>15p)",
    )
    parser.add_argument("--topic", type=str, default=None, help="Chủ đề video cụ thể (chạy video đơn lẻ)")
    parser.add_argument("--duration", type=int, default=None, help="Ghi đè thời lượng mục tiêu (giây)")
    parser.add_argument("--list-series", action="store_true", help="Xem danh sách tiến độ các chuỗi video trong DB")
    parser.add_argument("--plan-series", type=str, default=None, help="Chủ động nhờ AI lên kế hoạch chuỗi video theo chủ đề")
    parser.add_argument("--episodes", type=int, default=5, help="Số tập cho chuỗi video mới (mặc định 5)")
    args = parser.parse_args()

    if args.list_series:
        from .series_manager import print_series_table

        print_series_table()
        return

    if args.plan_series:
        from .series_manager import plan_new_series, print_series_table

        plan_new_series(theme=args.plan_series, num_episodes=args.episodes)
        print_series_table()
        return

    mode = apply_mode(args.mode)
    if args.duration:
        CONFIG["target_duration_seconds"] = int(args.duration)

    log.info(
        "Mode: %s (%dx%d, %ds) | Topic: %s",
        mode,
        CONFIG["visual"]["width"],
        CONFIG["visual"]["height"],
        CONFIG["target_duration_seconds"],
        args.topic or "Series Auto-pick",
    )

    n = int(CONFIG.get("videos_per_run", 1))
    for i in range(n):
        log.info("=== Video %d/%d ===", i + 1, n)
        run_once(topic=args.topic, upload_video=not args.no_upload, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
