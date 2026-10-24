"""SFX Generator: Synthesizes clean, high-precision sound effects for algorithm visualizations.

Creates:
  - assets/sfx/click.wav: Crisp mechanical switch click / tap.
  - assets/sfx/pop.wav: Soft bubble pop for cell/element highlight.
  - assets/sfx/swap.wav: Melodic marimba tone (frequency mapped to value).
  - assets/sfx/whoosh.wav: Smooth low-frequency transition sweep.
  - assets/sfx/complete.wav: Ascending 3-tone victory chime for sorted state.

Uses pure Python standard library (wave, struct, math) - zero external dependencies.
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

SFX_DIR = Path("assets/sfx")


def _write_wav(filepath: Path, samples: list[float], sample_rate: int = 44100) -> None:
    """Write mono 16-bit PCM WAV file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(filepath), "w") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        
        raw_data = bytearray()
        for s in samples:
            # Clamp to [-1.0, 1.0]
            val = max(-1.0, min(1.0, s))
            int_val = int(val * 32767.0)
            raw_data.extend(struct.pack("<h", int_val))
        wav_file.writeframes(raw_data)


def generate_click(sample_rate: int = 44100) -> list[float]:
    """Crisp mechanical switch click (~20ms)."""
    duration = 0.025
    total_samples = int(sample_rate * duration)
    samples = []
    for i in range(total_samples):
        t = i / sample_rate
        # Transient burst + exponential decay
        decay = math.exp(-t * 220.0)
        freq1 = 2800.0 + 800.0 * math.exp(-t * 300.0)
        freq2 = 1400.0
        val = (0.7 * math.sin(2.0 * math.pi * freq1 * t) + 0.3 * math.sin(2.0 * math.pi * freq2 * t)) * decay
        samples.append(val * 0.8)
    return samples


def generate_pop(sample_rate: int = 44100) -> list[float]:
    """Soft bubble pop sound (~40ms)."""
    duration = 0.04
    total_samples = int(sample_rate * duration)
    samples = []
    for i in range(total_samples):
        t = i / sample_rate
        decay = math.exp(-t * 120.0)
        # Pitch slides rapidly downwards
        freq = 600.0 * math.exp(-t * 60.0) + 180.0
        val = math.sin(2.0 * math.pi * freq * t) * decay
        samples.append(val * 0.75)
    return samples


def generate_swap(freq: float = 523.25, sample_rate: int = 44100) -> list[float]:
    """Marimba-like bell tone for element swap (~80ms). Pitch defaults to C5."""
    duration = 0.08
    total_samples = int(sample_rate * duration)
    samples = []
    for i in range(total_samples):
        t = i / sample_rate
        decay = math.exp(-t * 45.0)
        # Fundamental + 1st harmonic overtone
        val = (0.8 * math.sin(2.0 * math.pi * freq * t) + 0.2 * math.sin(2.0 * math.pi * freq * 2.75 * t)) * decay
        samples.append(val * 0.6)
    return samples


def generate_whoosh(sample_rate: int = 44100) -> list[float]:
    """Smooth low-frequency transition whoosh (~250ms)."""
    duration = 0.25
    total_samples = int(sample_rate * duration)
    samples = []
    for i in range(total_samples):
        t = i / sample_rate
        # Bell-shaped envelope
        env = math.sin(math.pi * (t / duration)) ** 2
        freq = 150.0 + 250.0 * (t / duration)
        val = math.sin(2.0 * math.pi * freq * t) * env
        samples.append(val * 0.5)
    return samples


def generate_complete(sample_rate: int = 44100) -> list[float]:
    """Ascending 3-chord victory chime (C5 -> E5 -> G5, ~350ms)."""
    notes = [523.25, 659.25, 783.99]  # C5, E5, G5
    note_dur = 0.11
    total_samples = int(sample_rate * (note_dur * len(notes) + 0.1))
    samples = [0.0] * total_samples
    
    for idx, f in enumerate(notes):
        start_idx = int(idx * note_dur * sample_rate)
        note_samples = int(sample_rate * 0.2)
        for i in range(note_samples):
            target_idx = start_idx + i
            if target_idx < total_samples:
                t = i / sample_rate
                decay = math.exp(-t * 20.0)
                val = (0.7 * math.sin(2.0 * math.pi * f * t) + 0.3 * math.sin(2.0 * math.pi * f * 2.0 * t)) * decay
                samples[target_idx] += val * 0.5
                
    return samples


def build_all_sfx() -> None:
    """Generate all standard algorithm visualization SFX."""
    SFX_DIR.mkdir(parents=True, exist_ok=True)
    _write_wav(SFX_DIR / "click.wav", generate_click())
    _write_wav(SFX_DIR / "pop.wav", generate_pop())
    _write_wav(SFX_DIR / "swap.wav", generate_swap(523.25))
    _write_wav(SFX_DIR / "whoosh.wav", generate_whoosh())
    _write_wav(SFX_DIR / "complete.wav", generate_complete())
    
    # Also generate a scale of 16 pitch-varied swap tones for element-value sonification
    pitches_dir = SFX_DIR / "pitches"
    pitches_dir.mkdir(parents=True, exist_ok=True)
    # Pentatonic scale from C4 (261Hz) to C6 (1046Hz)
    pentatonic_ratios = [1.0, 1.125, 1.25, 1.5, 1.667, 2.0, 2.25, 2.5, 3.0, 3.333, 4.0]
    base_freq = 261.63
    for i, ratio in enumerate(pentatonic_ratios):
        freq = base_freq * ratio
        _write_wav(pitches_dir / f"tone_{i:02d}.wav", generate_swap(freq))


if __name__ == "__main__":
    build_all_sfx()
    print("Generated all algorithm visualization SFX in assets/sfx/.")
