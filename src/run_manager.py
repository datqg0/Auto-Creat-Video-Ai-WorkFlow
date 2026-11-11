"""Idempotency key và checkpoint facade cho pipeline production."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import db


def make_run_key(mode: str, topic: str, episode_id: int | None = None) -> str:
    """Series retry ổn định theo episode; video đơn ổn định trong cùng ngày UTC."""
    identity = f"episode:{episode_id}" if episode_id else f"date:{datetime.now(timezone.utc).date().isoformat()}"
    raw = f"{mode.strip().lower()}|{identity}|{' '.join(topic.lower().split())}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class RunManager:
    run_key: str
    video_id: int
    workdir: Path

    @property
    def run_id(self) -> str:
        return self.run_key[:24]

    def start(self) -> None:
        self.workdir.mkdir(parents=True, exist_ok=True)
        db.init_pipeline_run(self.run_id, self.run_key, self.video_id)
        self._write_local("init", "running", {})

    def stage(self, name: str, status: str, output: dict[str, Any] | None = None, error: str | None = None) -> None:
        payload = output or {}
        serialized = json.dumps(payload, ensure_ascii=False, default=str)
        db.record_stage(self.run_id, name, status, output_json=serialized, error=error)
        self._write_local(name, status, payload, error)

    def complete(self, status: str = "completed") -> None:
        db.complete_pipeline_run(self.run_id, status)
        self._write_local("pipeline", status, {})

    def _write_local(self, stage: str, status: str, output: dict[str, Any], error: str | None = None) -> None:
        path = self.workdir / "run_state.json"
        state: dict[str, Any] = {"run_id": self.run_id, "run_key": self.run_key, "video_id": self.video_id, "stages": {}}
        if path.exists():
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        state.setdefault("stages", {})[stage] = {
            "status": status,
            "output": output,
            "error": error,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
