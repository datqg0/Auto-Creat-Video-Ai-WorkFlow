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


# Sample rate chuẩn khi phải chuyển đổi (Edge gốc 24kHz; VieNeu v3turbo 48kHz).
# KHÔNG hạ xuống 16kHz: giọng sẽ bị "đục" như qua điện thoại.
_TARGET_SR = 24000


def _concat_wavs(parts: list[Path], out: Path) -> None:
    """Ghép nhiều file wav thành 1 file.

    Cùng định dạng -> ghép thẳng bằng ``wave``. Khác định dạng (vd. lỡ fallback
    provider giữa chừng) -> nhờ ffmpeg resample về chung một chuẩn rồi ghép.
    """
    params_list = []
    for p in parts:
        with wave.open(str(p), "rb") as w:
            params_list.append(w.getparams()[:3])  # nchannels, sampwidth, framerate
    if all(pr == params_list[0] for pr in params_list):
        with wave.open(str(parts[0]), "rb") as first:
            params = first.getparams()
        with wave.open(str(out), "wb") as w:
            w.setparams(params)
            for p in parts:
                with wave.open(str(p), "rb") as seg:
                    w.writeframes(seg.readframes(seg.getnframes()))
        return

    import subprocess

    sr = max(pr[2] for pr in params_list)
    cmd = ["ffmpeg", "-y"]
    for p in parts:
        cmd += ["-i", str(p)]
    inputs = "".join(f"[{i}:a]aresample={sr},aformat=channel_layouts=mono[a{i}];" for i in range(len(parts)))
    joined = "".join(f"[a{i}]" for i in range(len(parts)))
    cmd += [
        "-filter_complex", f"{inputs}{joined}concat=n={len(parts)}:v=0:a=1[out]",
        "-map", "[out]", "-c:a", "pcm_s16le", str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


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
    speed = float(cfg.get("speed", 0.95))
    try:
        audio = engine.infer(text, voice=voice, speed=speed)
    except TypeError:
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
        output_format="pcm_24000",
    )
    # Ghi PCM thô thành WAV
    pcm = b"".join(audio)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(pcm)


# ---------- Edge-TTS ----------
def _synth_edge(text: str, out: Path) -> None:
    import edge_tts

    cfg = CONFIG["tts"]["edge"]
    lang = CONFIG.get("language", "vi")
    voice = cfg["voice_vi"] if lang == "vi" else cfg["voice_en"]
    rate = str(cfg.get("rate", "-6%"))
    pitch = str(cfg.get("pitch", "+0Hz"))
    mp3_path = out.with_suffix(".mp3")

    async def _run() -> None:
        communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
        await communicate.save(str(mp3_path))

    asyncio.run(_run())
    # Chuyển mp3 -> wav mono, GIỮ 24kHz gốc của Edge (16kHz làm giọng đục).
    import subprocess

    subprocess.run(
        ["ffmpeg", "-y", "-i", str(mp3_path), "-ar", str(_TARGET_SR), "-ac", "1", str(out)],
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


def clean_tts_text(text: str) -> str:
    """Loại bỏ triệt để ký tự markdown, emoji, bullet, URL, ký tự đặc biệt để giọng TTS phát âm chuẩn nhất."""
    if not text:
        return ""
    import re

    # Bỏ link web URL nếu có lọt vào
    t = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Bỏ markdown links: [text](url) -> text
    t = re.sub(r"\[([^\]]+)\]\([^\)]*\)", r"\1", t)
    # Bỏ markdown bold/italic: ***text***, **text**, *text*, ___text___, __text__, _text_
    t = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", t)
    t = re.sub(r"_{1,3}(.*?)_{1,3}", r"\1", t)
    t = re.sub(r"~~(.*?)~~", r"\1", t)
    t = re.sub(r"`+([^`]+)`+", r"\1", t)
    # Bỏ emoji / unicode biểu tượng (🔥, 🚀, 👋, 💡...)
    t = re.sub(r"[\U00010000-\U0010ffff]", "", t)
    # Bỏ các ký hiệu gây đọc lạ: bullet •, hashtag #, sao *, gạch dưới _, ngã ~, backtick `, gạch đứng |, gạch chéo \, mũ ^, ngoặc nhọn, ngoặc vuông
    t = re.sub(r"[•#\*\_~`|\\^<>{}\[\]]", " ", t)
    # Bỏ gạch nối đơn độc (tránh đọc thành 'trừ' hoặc 'gạch')
    t = re.sub(r"(?:^|\s)[-\u2013\u2014]+(?:\s|$)", " ", t)
    # Chuẩn hóa khoảng trắng
    return re.sub(r"\s+", " ", t).strip()


def synthesize(text: str, out_path: Path) -> float:
    """Đọc text ra file wav. Trả về thời lượng (giây).

    Lần đầu: thử lần lượt provider theo config, KHÓA vào provider đầu tiên chạy được.
    Các lần sau: chỉ dùng provider đã khóa để giữ NGUYÊN một giọng cho cả video.
    """
    global _LOCKED_PROVIDER
    text = clean_tts_text(text)
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


def _append_silence(path: Path, duration: float = 0.25) -> None:
    """Nối thêm một khoảng lặng ngắn vào cuối file wav để tạo nhịp thở tự nhiên."""
    if duration <= 0:
        return
    try:
        with wave.open(str(path), "rb") as r:
            params = r.getparams()
            frames = r.readframes(r.getnframes())
        silence_bytes = b"\x00" * int(params.framerate * params.sampwidth * params.nchannels * duration)
        with wave.open(str(path), "wb") as w:
            w.setparams(params)
            w.writeframes(frames)
            w.writeframes(silence_bytes)
    except Exception as e:
        log.debug("Thêm khoảng lặng wav lỗi: %s", e)


def synthesize_timed(text: str, out_path: Path) -> list[tuple[str, float]]:
    """Đọc text TỪNG CÂU rồi ghép thành 1 file wav; trả về [(câu, thời lượng)].

    Biết chính xác mỗi câu bắt đầu/kết thúc lúc nào -> phụ đề (và hiệu ứng nhấn
    sau này) khớp giọng đọc, thay vì chia đều theo số ký tự rồi lệch dần.
    Mỗi câu vẫn đi qua ``synthesize`` nên giữ nguyên cơ chế khóa giọng/fallback.
    """
    text = clean_tts_text(text)
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        dur = synthesize(text, out_path)
        return [(text.strip(), dur)]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pause_sec = float(CONFIG.get("tts", {}).get("sentence_pause", 0.25))
    parts: list[Path] = []
    timings: list[tuple[str, float]] = []
    try:
        for j, sent in enumerate(sentences):
            part = out_path.with_name(f"{out_path.stem}_s{j:02d}.wav")
            dur = synthesize(sent, part)
            # Chèn khoảng nghỉ ngắn giữa các câu (trừ câu cuối của scene)
            if pause_sec > 0 and j < len(sentences) - 1:
                _append_silence(part, pause_sec)
                dur += pause_sec
            parts.append(part)
            timings.append((sent, dur))
        _concat_wavs(parts, out_path)
    finally:
        for p in parts:
            p.unlink(missing_ok=True)

    # Chuẩn hóa tổng thời lượng theo file đã ghép (sai số resample nếu có).
    total = _wav_duration(out_path)
    raw = sum(d for _, d in timings) or 1.0
    return [(s, d * total / raw) for s, d in timings]
