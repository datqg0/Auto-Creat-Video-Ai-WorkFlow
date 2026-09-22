"""Sinh phụ đề .srt từ file audio bằng faster-whisper."""
from __future__ import annotations

import logging
from pathlib import Path

from .config import CONFIG

log = logging.getLogger(__name__)


def _fmt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def generate_srt(audio_path: Path, srt_path: Path) -> Path:
    """Nghe audio -> tạo .srt căn timestamp. Trả về đường dẫn .srt."""
    from faster_whisper import WhisperModel

    cfg = CONFIG["subtitles"]
    model_size = cfg.get("whisper_model", "base")
    lang = CONFIG.get("language", "vi")

    log.info("Whisper (%s) sinh phụ đề cho %s", model_size, audio_path.name)
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(audio_path), language=lang, vad_filter=True)

    srt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            f.write(f"{i}\n")
            f.write(f"{_fmt_ts(seg.start)} --> {_fmt_ts(seg.end)}\n")
            f.write(f"{seg.text.strip()}\n\n")
    return srt_path
