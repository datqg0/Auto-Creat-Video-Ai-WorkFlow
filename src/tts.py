"""TTS với fallback: VieNeu-TTS -> ElevenLabs -> Edge-TTS.

Mỗi scene được đọc riêng thành 1 file wav để đồng bộ với visual.
Trả về đường dẫn file audio + thời lượng (giây).
"""
from __future__ import annotations

import asyncio
import logging
import wave
from pathlib import Path

from .config import CONFIG, env

log = logging.getLogger(__name__)


class TTSError(RuntimeError):
    pass


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def _concat_wavs(parts: list[Path], out: Path) -> None:
    """Ghép nhiều file wav cùng định dạng thành 1 file."""
    with wave.open(str(parts[0]), "rb") as first:
        params = first.getparams()
    with wave.open(str(out), "wb") as w:
        w.setparams(params)
        for p in parts:
            with wave.open(str(p), "rb") as seg:
                w.writeframes(seg.readframes(seg.getnframes()))


def _split_sentences(text: str) -> list[str]:
    """Cắt narration thành câu ngắn để đọc lại từng phần khi 1 lần đọc bị lỗi."""
    import re

    chunks = re.split(r"(?<=[.!?…])\s+", text.strip())
    chunks = [c.strip() for c in chunks if c.strip()]
    return chunks or [text.strip()]


# ---------- VieNeu-TTS ----------
_VIENEU_ENGINE = None


def _get_vieneu():
    """Khởi tạo engine VieNeu 1 lần rồi tái dùng (load model tốn ~20s)."""
    global _VIENEU_ENGINE
    if _VIENEU_ENGINE is None:
        from vieneu import Vieneu  # type: ignore

        cfg = CONFIG["tts"].get("vieneu", {})
        _VIENEU_ENGINE = Vieneu(mode=cfg.get("mode", "v3turbo"))
    return _VIENEU_ENGINE


def _synth_vieneu(text: str, out: Path) -> None:
    try:
        engine = _get_vieneu()
    except Exception as e:  # noqa: BLE001
        raise TTSError(f"VieNeu-TTS chưa cài được: {e}")

    cfg = CONFIG["tts"].get("vieneu", {})
    voice = cfg.get("voice", "Minh Quân Pro")
    audio = engine.infer(text, voice=voice)
    engine.save(audio, str(out))


# ---------- ElevenLabs ----------
def _synth_elevenlabs(text: str, out: Path) -> None:
    api_key = env("ELEVENLABS_API_KEY")
    if not api_key:
        raise TTSError("Thiếu ELEVENLABS_API_KEY")
    from elevenlabs.client import ElevenLabs

    cfg = CONFIG["tts"]["elevenlabs"]
    client = ElevenLabs(api_key=api_key)
    audio = client.text_to_speech.convert(
        voice_id=cfg["voice_id"],
        model_id=cfg.get("model", "eleven_multilingual_v2"),
        text=text,
        output_format="pcm_16000",
    )
    # Ghi PCM thô thành WAV
    pcm = b"".join(audio)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm)


# ---------- Edge-TTS ----------
def _synth_edge(text: str, out: Path) -> None:
    import edge_tts

    cfg = CONFIG["tts"]["edge"]
    lang = CONFIG.get("language", "vi")
    voice = cfg["voice_vi"] if lang == "vi" else cfg["voice_en"]
    mp3_path = out.with_suffix(".mp3")

    async def _run() -> None:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(mp3_path))

    asyncio.run(_run())
    # Chuyển mp3 -> wav 16k mono bằng ffmpeg
    import subprocess

    subprocess.run(
        ["ffmpeg", "-y", "-i", str(mp3_path), "-ar", "16000", "-ac", "1", str(out)],
        check=True,
        capture_output=True,
    )
    mp3_path.unlink(missing_ok=True)


_DISPATCH = {
    "vieneu": _synth_vieneu,
    "elevenlabs": _synth_elevenlabs,
    "edge": _synth_edge,
}

# Khóa provider cho cả video: scene đầu chọn được provider nào thì mọi scene
# sau CHỈ dùng đúng provider đó -> giọng đồng nhất, không bị lẫn giọng giữa video.
_LOCKED_PROVIDER: str | None = None


def reset_provider_lock() -> None:
    """Bỏ khóa provider (gọi ở đầu mỗi video để chọn lại từ đầu)."""
    global _LOCKED_PROVIDER
    _LOCKED_PROVIDER = None


def _try_provider(name: str, text: str, out_path: Path) -> bool:
    fn = _DISPATCH.get(name)
    if not fn:
        return False
    log.info("TTS %s: %s...", name, text[:40])
    fn(text, out_path)
    return out_path.exists() and out_path.stat().st_size > 0


def _synth_by_sentences(name: str, text: str, out_path: Path) -> bool:
    """Đọc từng câu bằng CÙNG provider rồi ghép lại.

    Dùng khi đọc cả đoạn dài bị lỗi: giữ NGUYÊN giọng thay vì đổi sang provider
    khác (tránh video bị lẫn giọng ở giữa/cuối).
    """
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return False
    parts: list[Path] = []
    for j, sent in enumerate(sentences):
        part = out_path.with_name(f"{out_path.stem}_p{j:02d}.wav")
        if not _try_provider(name, sent, part):
            for p in parts:
                p.unlink(missing_ok=True)
            return False
        parts.append(part)
    _concat_wavs(parts, out_path)
    for p in parts:
        p.unlink(missing_ok=True)
    return out_path.exists() and out_path.stat().st_size > 0


def synthesize(text: str, out_path: Path) -> float:
    """Đọc text ra file wav. Trả về thời lượng (giây).

    Lần đầu: thử lần lượt provider theo config, KHÓA vào provider đầu tiên chạy được.
    Các lần sau: chỉ dùng provider đã khóa để giữ NGUYÊN một giọng cho cả video.
    """
    global _LOCKED_PROVIDER
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Đã khóa provider -> chỉ dùng đúng nó để giọng không đổi.
    if _LOCKED_PROVIDER is not None:
        try:
            if _try_provider(_LOCKED_PROVIDER, text, out_path):
                return _wav_duration(out_path)
        except Exception as e:  # noqa: BLE001
            log.warning(
                "TTS %s (đã khóa) đọc cả đoạn lỗi: %s — thử đọc từng câu cùng giọng",
                _LOCKED_PROVIDER, e,
            )
        # Đọc cả đoạn hỏng -> thử đọc TỪNG CÂU bằng đúng giọng đã khóa (giữ giọng).
        try:
            if _synth_by_sentences(_LOCKED_PROVIDER, text, out_path):
                return _wav_duration(out_path)
        except Exception as e:  # noqa: BLE001
            log.warning(
                "TTS %s (đã khóa) đọc từng câu vẫn lỗi: %s — buộc phải fallback giọng khác",
                _LOCKED_PROVIDER, e,
            )

    last_err: Exception | None = None
    for name in CONFIG["tts"]["providers"]:
        # Nếu đã khóa và vừa thử thất bại ở trên thì bỏ qua provider đã khóa.
        if name == _LOCKED_PROVIDER and _LOCKED_PROVIDER is not None:
            continue
        try:
            if _try_provider(name, text, out_path):
                if _LOCKED_PROVIDER is None:
                    _LOCKED_PROVIDER = name
                    log.info("TTS khóa provider cho cả video: %s", name)
                return _wav_duration(out_path)
        except Exception as e:  # noqa: BLE001
            last_err = e
            log.warning("TTS %s lỗi: %s", name, e)
    raise TTSError(f"Tất cả TTS provider đều lỗi. Cuối: {last_err}")
