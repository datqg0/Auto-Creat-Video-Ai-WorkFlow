"""Test tích hợp: render 1 video ngắn có scene động (mathviz) qua compositor.

Tạo audio giả (im lặng) đúng thời lượng để không phụ thuộc TTS/LLM, rồi ghép
1 scene tĩnh + 1 scene động qua compose(). Chạy:
    python -m src.anim_integration_test
"""
from __future__ import annotations

import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import Scene  # noqa: E402
from src.compositor import compose  # noqa: E402
from src.visual_engine import render_scene  # noqa: E402
from src.animation_bridge import build_animation_scene  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "output" / "anim_integration"
OUT.mkdir(parents=True, exist_ok=True)


def _silence(path: Path, seconds: float, rate: int = 16000) -> None:
    n = int(seconds * rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * n)


def main() -> None:
    scenes = [
        Scene(
            narration="Giới thiệu.",
            visual_type="title",
            heading="Hàm sin và mạng neural",
        ),
        Scene(
            narration="Đồ thị hàm sin.",
            visual_type="animation",
            heading="y = sin(x)",
            animation={"preset": "function", "expr": "sin(x)"},
        ),
        Scene(
            narration="Mạng neural.",
            visual_type="animation",
            heading="Mạng Neural",
            animation={"preset": "neural_net", "layers": [3, 5, 4, 2]},
        ),
    ]

    images, audios, anims = [], [], []
    for i, sc in enumerate(scenes):
        img = OUT / f"scene_{i:02d}.png"
        aud = OUT / f"scene_{i:02d}.wav"
        render_scene(sc, img)
        _silence(aud, 3.5)
        images.append(img)
        audios.append(aud)
        anim = None
        if sc.visual_type == "animation" and sc.animation:
            anim = build_animation_scene(sc, 3.5)
        anims.append(anim)

    out = compose(images, audios, OUT / "video.mp4", None, anims)
    print(f"VIDEO -> {out}")


if __name__ == "__main__":
    main()
