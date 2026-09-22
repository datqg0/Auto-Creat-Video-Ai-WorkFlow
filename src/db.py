"""SQLite lưu lịch sử chủ đề đã dùng và trạng thái video."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "output" / "state.db"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                title TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                youtube_id TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def recent_topics(days: int) -> list[str]:
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT topic FROM videos WHERE created_at >= ?", (cutoff,)
        ).fetchall()
    return [r["topic"] for r in rows]


def create_video(topic: str) -> int:
    now = datetime.utcnow().isoformat()
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO videos (topic, status, created_at, updated_at) "
            "VALUES (?, 'pending', ?, ?)",
            (topic, now, now),
        )
        conn.commit()
        return int(cur.lastrowid)


def update_video(video_id: int, **fields: str) -> None:
    if not fields:
        return
    fields["updated_at"] = datetime.utcnow().isoformat()
    cols = ", ".join(f"{k} = ?" for k in fields)
    with _connect() as conn:
        conn.execute(
            f"UPDATE videos SET {cols} WHERE id = ?",
            (*fields.values(), video_id),
        )
        conn.commit()
