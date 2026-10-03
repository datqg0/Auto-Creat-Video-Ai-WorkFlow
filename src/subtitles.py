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
    scene_texts: list[str],
    scene_durations: list[float],
    srt_path: Path,
    scene_timings: list[list[tuple[str, float]] | None] | None = None,
) -> Path:
    """Tạo .srt từ text gốc + thời lượng mỗi scene.

    Nếu có ``scene_timings`` (mốc THẬT của từng câu do TTS đọc riêng) thì căn theo
    từng câu; tỉ lệ ký tự chỉ dùng để chia nhỏ BÊN TRONG một câu -> sai số không
    cộng dồn. Không có -> chia cả scene theo tỉ lệ ký tự như cũ.
    """
    srt_path.parent.mkdir(parents=True, exist_ok=True)
    idx = 1
    t = 0.0
    lines: list[str] = []
    if scene_timings is None:
        scene_timings = [None] * len(scene_texts)

    for text, dur, timing in zip(scene_texts, scene_durations, scene_timings):
        # Danh sách (đoạn text, thời lượng) cấp câu cho scene này.
        groups = timing if timing else [(text, dur)]
        start = t
        for sent, sent_dur in groups:
            chunks = _split_sentences(sent)
            total_chars = sum(len(c) for c in chunks) or 1
            for c in chunks:
                seg = sent_dur * (len(c) / total_chars)
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

    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            f.write(f"{i}\n")
            f.write(f"{_fmt_ts(seg.start)} --> {_fmt_ts(seg.end)}\n")
            f.write(f"{seg.text.strip()}\n\n")
    return srt_path


def translate_srt_to_english(vi_srt_path: Path, en_srt_path: Path) -> Path | None:
    """Đọc file .srt tiếng Việt, dịch sang tiếng Anh bằng LLM, giữ nguyên timestamp."""
    if not vi_srt_path.exists():
        return None
    content = vi_srt_path.read_text(encoding="utf-8").strip()
    if not content:
        return None

    import re
    blocks = [b.strip() for b in content.split("\n\n") if b.strip()]
    texts_to_translate: list[str] = []
    headers: list[tuple[str, str]] = []
    for b in blocks:
        lines = b.split("\n")
        if len(lines) >= 3:
            headers.append((lines[0], lines[1]))
            texts_to_translate.append(" ".join(lines[2:]))

    if not texts_to_translate:
        return None

    try:
        from .llm import generate_text

        prompt = (
            "You are a professional subtitle translator.\n"
            "Translate the following numbered Vietnamese subtitle lines into natural, concise English.\n"
            "Keep the exact same numbering and line count.\n"
            "Do NOT add commentary. Return ONLY the translated lines.\n\n"
            + "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts_to_translate))
        )
        translated_raw = generate_text(prompt, max_tokens=3000)
        raw_lines = [l.strip() for l in translated_raw.strip().split("\n") if l.strip()]

        en_lines: list[str] = []
        for line in raw_lines:
            m = re.match(r"^\d+[\.\:\)\-]\s*(.*)$", line)
            en_lines.append(m.group(1).strip() if m else line)

        out_blocks: list[str] = []
        for i, (idx, ts) in enumerate(headers):
            text_en = en_lines[i] if i < len(en_lines) else texts_to_translate[i]
            out_blocks.append(f"{idx}\n{ts}\n{text_en}")

        en_srt_path.parent.mkdir(parents=True, exist_ok=True)
        en_srt_path.write_text("\n\n".join(out_blocks) + "\n\n", encoding="utf-8")
        log.info("Đã tạo phụ đề tiếng Anh: %s", en_srt_path.name)
        return en_srt_path
    except Exception as e:  # noqa: BLE001
        log.warning("Dịch phụ đề tiếng Anh lỗi: %s", e)
        return None
