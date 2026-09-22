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


def synthesize(text: str, out_path: Path) -> float:
    """Đọc text ra file wav, thử lần lượt các provider. Trả về thời lượng (giây)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for name in CONFIG["tts"]["providers"]:
        fn = _DISPATCH.get(name)
        if not fn:
            continue
        try:
            log.info("TTS %s: %s...", name, text[:40])
            fn(text, out_path)
            if out_path.exists() and out_path.stat().st_size > 0:
                return _wav_duration(out_path)
        except Exception as e:  # noqa: BLE001
            last_err = e
            log.warning("TTS %s lỗi: %s", name, e)
    raise TTSError(f"Tất cả TTS provider đều lỗi. Cuối: {last_err}")
