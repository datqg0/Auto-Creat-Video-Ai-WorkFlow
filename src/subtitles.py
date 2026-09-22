"""Sinh phụ đề .srt.

Cách chuẩn xác nhất: dùng chính text narration gốc (đã biết 100%) + thời lượng
audio mỗi scene, chia theo số ký tự -> timestamp. Không đoán sai như Whisper.

Vẫn giữ generate_srt (Whisper) làm phương án dự phòng.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from .config import CONFIG

log = logging.getLogger(__name__)


def _fmt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _split_sentences(text: str) -> list[str]:
    """Tách narration thành cụm ngắn để hiện phụ đề (theo dấu câu, rồi theo độ dài)."""
    text = " ".join(text.split())
    parts = re.split(r"(?<=[.!?…:;])\s+|(?<=,)\s+", text)
    chunks: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        while len(p) > 48:
            cut = p.rfind(" ", 0, 48)
            if cut <= 0:
                cut = 48
            chunks.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            chunks.append(p)
    return chunks or [text]


def srt_from_scenes(
    scene_texts: list[str], scene_durations: list[float], srt_path: Path
) -> Path:
    """Tạo .srt từ text gốc + thời lượng mỗi scene. Timestamp căn theo tỉ lệ ký tự."""
    srt_path.parent.mkdir(parents=True, exist_ok=True)
    idx = 1
    t = 0.0
    lines: list[str] = []

    for text, dur in zip(scene_texts, scene_durations):
        chunks = _split_sentences(text)
        total_chars = sum(len(c) for c in chunks) or 1
        start = t
        for c in chunks:
            seg = dur * (len(c) / total_chars)
            end = start + seg
            lines.append(str(idx))
            lines.append(f"{_fmt_ts(start)} --> {_fmt_ts(end)}")
            lines.append(c)
            lines.append("")
            idx += 1
            start = end
        t += dur

    srt_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Phụ đề (text gốc) -> %s (%d dòng)", srt_path.name, idx - 1)
    return srt_path


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
