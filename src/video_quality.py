"""Pre-upload render checks and visual review artifacts."""
from __future__ import annotations

import json
import logging
import math
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .config import CONFIG
from .models import Script
from .script_quality import inspect_script

log = logging.getLogger(__name__)


def _add(checks: list[dict], name: str, ok: bool, message: str, severity: str = "error") -> None:
    checks.append({"name": name, "ok": bool(ok), "severity": severity, "message": message})


def create_contact_sheet(workdir: Path, output_path: Path, max_columns: int = 4) -> Path | None:
    """Combine rendered scene stills into a compact storyboard for human review."""
    paths = sorted(workdir.glob("scene_[0-9][0-9].png"))
    paths += sorted(workdir.glob("aivideo_*_frames/frame_*.jpg"))
    if not paths:
        return None
    tile_w, tile_h, label_h, pad = 360, 203, 34, 12
    columns = min(max_columns, len(paths))
    rows = math.ceil(len(paths) / columns)
    sheet = Image.new("RGB", (columns * (tile_w + pad) + pad, rows * (tile_h + label_h + pad) + pad), "#0d1117")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, path in enumerate(paths):
        x = pad + (index % columns) * (tile_w + pad)
        y = pad + (index // columns) * (tile_h + label_h + pad)
        try:
            with Image.open(path) as src:
                tile = ImageOps.fit(src.convert("RGB"), (tile_w, tile_h), method=Image.Resampling.LANCZOS)
            sheet.paste(tile, (x, y))
            label = path.name if path.parent == workdir else f"{path.parent.name}/{path.name}"
            draw.text((x + 4, y + tile_h + 8), label[:48], fill="#e6edf3", font=font)
        except Exception as exc:  # noqa: BLE001
            draw.text((x + 4, y + 8), f"{path.name}: {exc}", fill="#ff7b72", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=88)
    return output_path


def _ffprobe(path: Path, timeout: int) -> dict:
    executable = shutil.which("ffprobe")
    if not executable:
        raise RuntimeError("Không tìm thấy ffprobe; cài FFmpeg đầy đủ để kiểm định video.")
    result = subprocess.run(
        [executable, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )
    return json.loads(result.stdout)


def _rate(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    try:
        numerator, denominator = value.split("/", 1)
        return float(numerator) / float(denominator)
    except (ValueError, ZeroDivisionError):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


def _subtitle_checks(path: Path | None, checks: list[dict], max_cps: float, max_chars: int) -> None:
    if path is None or not path.exists():
        _add(checks, "captions", False, "Không có file phụ đề tiếng Việt.", "warning")
        return
    content = path.read_text(encoding="utf-8-sig", errors="replace").strip()
    cues = [block for block in re.split(r"\n\s*\n", content) if block.strip()]
    malformed = 0
    overlaps = 0
    too_fast = 0
    too_long = 0
    previous_end = 0.0
    time_re = re.compile(
        r"(?P<start>\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(?P<end>\d{2}:\d{2}:\d{2}[,.]\d{3})"
    )

    def seconds(ts: str) -> float:
        h, m, rest = ts.replace(",", ".").split(":")
        return int(h) * 3600 + int(m) * 60 + float(rest)

    for block in cues:
        lines = block.splitlines()
        match = next((time_re.search(line) for line in lines if "-->" in line), None)
        if not match:
            malformed += 1
            continue
        start, end = seconds(match.group("start")), seconds(match.group("end"))
        texts = [line.strip() for line in lines if line.strip() and "-->" not in line and not line.strip().isdigit()]
        text = " ".join(texts)
        duration = end - start
        if duration <= 0:
            malformed += 1
            continue
        if start < previous_end - 0.05:
            overlaps += 1
        previous_end = max(previous_end, end)
        if len(text) > max_chars:
            too_long += 1
        if text and len(text) / duration > max_cps:
            too_fast += 1

    _add(checks, "caption_timestamps", malformed == 0 and overlaps == 0,
         f"{len(cues)} cue; lỗi thời gian {malformed}, chồng lấn {overlaps}.")
    _add(checks, "caption_readability", too_fast == 0 and too_long == 0,
         f"{too_fast} cue đọc quá nhanh, {too_long} cue quá dài.", "warning")


def _scan_black_and_freeze(path: Path, checks: list[dict], timeout: int) -> None:
    executable = shutil.which("ffmpeg")
    if not executable:
        _add(checks, "visual_scan", False, "Không tìm thấy ffmpeg để quét khung đen/đứng hình.", "warning")
        return
    try:
        result = subprocess.run(
            [
                executable, "-hide_banner", "-i", str(path), "-vf",
                "blackdetect=d=2:pix_th=0.08,freezedetect=n=-50dB:d=8",
                "-an", "-f", "null", "-",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        log_text = (result.stderr or "")
        black_count = len(re.findall(r"black_duration:[0-9.]+", log_text))
        freeze_count = len(re.findall(r"freeze_duration:[0-9.]+", log_text))
        if result.returncode:
            _add(checks, "visual_scan", False, f"FFmpeg không hoàn tất quét: {log_text[-500:]}", "warning")
            return
        ok = black_count == 0 and freeze_count == 0
        _add(checks, "visual_scan", ok,
             f"Phát hiện {black_count} đoạn đen dài và {freeze_count} đoạn đứng hình dài.", "warning")
    except Exception as exc:  # noqa: BLE001
        _add(checks, "visual_scan", False, f"Không quét được video: {exc}", "warning")


def audit_render(
    video_path: Path,
    thumbnail_path: Path,
    workdir: Path,
    script: Script,
    expected_duration: float,
    scene_renderers: list[str] | None = None,
) -> dict:
    """Write a deterministic QA report; only critical delivery defects block upload."""
    cfg = CONFIG.get("quality_control", {}) or {}
    checks: list[dict] = []
    script_report = inspect_script(script)
    _add(checks, "script_integrity", script_report["passed"],
         f"{len(script_report['errors'])} lỗi nội dung kiểm định.")
    for message in script_report["errors"]:
        _add(checks, "script_error", False, message)
    for message in script_report["warnings"]:
        _add(checks, "script_review", False, message, "warning")

    contact_sheet = create_contact_sheet(workdir, workdir / "scene_contact_sheet.jpg")
    _add(checks, "storyboard", contact_sheet is not None,
         str(contact_sheet.name if contact_sheet else "Không tìm thấy ảnh cảnh."), "warning")

    if not video_path.exists() or video_path.stat().st_size < int(cfg.get("min_file_bytes", 50000)):
        _add(checks, "video_file", False, "Video thiếu hoặc nhỏ bất thường.")
        report = _make_report(checks, script_report, contact_sheet, None, scene_renderers)
        _write_report(workdir, report)
        return report
    _add(checks, "video_file", True, f"{video_path.stat().st_size:,} bytes.")

    probe = None
    try:
        probe = _ffprobe(video_path, int(cfg.get("probe_timeout_seconds", 900)))
    except Exception as exc:  # noqa: BLE001
        _add(checks, "ffprobe", False, str(exc))

    duration = None
    if probe:
        streams = probe.get("streams", [])
        video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
        audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
        _add(checks, "video_stream", video is not None, "Có video stream." if video else "Thiếu video stream.")
        _add(checks, "audio_stream", audio is not None, "Có audio stream." if audio else "Thiếu audio stream.")
        try:
            duration = float((probe.get("format") or {}).get("duration") or (video or {}).get("duration"))
        except (TypeError, ValueError):
            duration = None
        _add(checks, "duration", duration is not None and duration > 0,
             f"Thời lượng {duration:.2f}s." if duration is not None else "Không đọc được thời lượng.")
        if duration is not None and expected_duration > 0:
            allowed_delta = max(float(cfg.get("duration_tolerance_seconds", 2.0)), expected_duration * 0.02)
            _add(checks, "duration_match", abs(duration - expected_duration) <= allowed_delta,
                 f"Video {duration:.2f}s; audio cảnh {expected_duration:.2f}s; lệch {abs(duration - expected_duration):.2f}s.")
        if video:
            expected_w = int(CONFIG.get("visual", {}).get("width", 1920))
            expected_h = int(CONFIG.get("visual", {}).get("height", 1080))
            actual_w, actual_h = int(video.get("width", 0)), int(video.get("height", 0))
            _add(checks, "resolution", actual_w == expected_w and actual_h == expected_h,
                 f"{actual_w}x{actual_h}; yêu cầu {expected_w}x{expected_h}.")
            fps = _rate(video.get("avg_frame_rate") or video.get("r_frame_rate"))
            expected_fps = float(CONFIG.get("visual", {}).get("fps", 30))
            _add(checks, "frame_rate", fps is not None and abs(fps - expected_fps) < 0.1,
                 f"{fps:.3f} fps; yêu cầu {expected_fps:g} fps." if fps is not None else "Không đọc được frame rate.")
        if audio:
            sample_rate = int(audio.get("sample_rate", 0) or 0)
            _add(checks, "audio_sample_rate", sample_rate == 48000,
                 f"Audio {sample_rate} Hz; yêu cầu 48000 Hz.")

    thumb_ok = False
    try:
        with Image.open(thumbnail_path) as thumb:
            thumb_ok = thumb.size == (1280, 720)
            thumb_size = thumb.size
    except Exception:
        thumb_size = None
    _add(checks, "thumbnail", thumb_ok,
         f"Thumbnail {thumb_size}; yêu cầu 1280x720.")

    is_vertical = int(CONFIG.get("visual", {}).get("height", 1080)) > int(CONFIG.get("visual", {}).get("width", 1920))
    _subtitle_checks(
        workdir / "subs_vi.srt" if (workdir / "subs_vi.srt").exists() else None,
        checks,
        float(cfg.get("max_caption_chars_per_second", 22)),
        int(cfg.get("max_caption_chars_vertical" if is_vertical else "max_caption_chars_horizontal", 32 if is_vertical else 48)),
    )
    renderers = scene_renderers or []
    if renderers:
        counts = Counter(renderers)
        static_ratio = counts.get("static", 0) / len(renderers)
        _add(checks, "scene_visual_mix", static_ratio <= float(cfg.get("max_static_scene_ratio", 0.4)),
             f"{counts.get('static', 0)}/{len(renderers)} cảnh dùng ảnh/zoom; renderer: {dict(counts)}.", "warning")

    if cfg.get("scan_black_and_freeze", True):
        _scan_black_and_freeze(video_path, checks, int(cfg.get("visual_scan_timeout_seconds", 1800)))

    report = _make_report(checks, script_report, contact_sheet, probe, renderers)
    _write_report(workdir, report)
    return report


def _make_report(checks: list[dict], script_report: dict, contact_sheet: Path | None, probe: dict | None,
                 scene_renderers: list[str] | None) -> dict:
    blocking = [c for c in checks if not c["ok"] and c["severity"] == "error"]
    warnings = [c for c in checks if not c["ok"] and c["severity"] == "warning"]
    cfg = CONFIG.get("quality_control", {}) or {}
    return {
        "passed": not blocking and (not warnings or not cfg.get("fail_on_warnings", False)),
        "blocking_errors": blocking,
        "warnings": warnings,
        "checks": checks,
        "script": script_report,
        "scene_renderers": scene_renderers or [],
        "contact_sheet": str(contact_sheet) if contact_sheet else None,
        "probe": probe,
    }


def _write_report(workdir: Path, report: dict) -> Path:
    path = workdir / "quality_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def require_quality_pass(report: dict, workdir: Path) -> None:
    """Stop public upload when the final render fails a blocking quality check."""
    for check in report.get("checks", []):
        if check.get("ok"):
            continue
        method = log.warning if check.get("severity") == "warning" else log.error
        method("Video QA [%s]: %s", check.get("name"), check.get("message"))
    if not report.get("passed"):
        raise RuntimeError(f"Video không qua quality gate; xem {workdir / 'quality_report.json'}")
    log.info("Video qua quality gate; báo cáo: %s", workdir / "quality_report.json")
