"""Database Adapter: Hỗ trợ SQLite Local và Neon (Serverless PostgreSQL).

Kiến trúc Fallback 4 tầng:
  1. Primary: Neon Serverless PostgreSQL (nếu cấu hình DATABASE_URL).
  2. Fallback 1 (Cloud -> Local): Nếu Neon lỗi kết nối/timeout/sai pass/down -> Tự động chuyển sang SQLite local (output/state.db).
  3. Fallback 2 (File -> Memory): Nếu file SQLite bị khóa hoặc lỗi disk -> Tự động chuyển sang SQLite in-memory (:memory:).
  4. Fallback 3 (Graceful Resilience): Ghi vết dự phòng vào output/topics_history.txt; các thao tác DB được bọc an toàn
     để KHÔNG BAO GIỜ làm gián đoạn quá trình render video hay upload YouTube.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .config import env

log = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "output" / "state.db"
TOPICS_FILE = Path(__file__).resolve().parent.parent / "output" / "topics_history.txt"

# Trạng thái fallback runtime
_force_sqlite: bool = False
_mem_conn: sqlite3.Connection | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_postgres() -> bool:
    """Kiểm tra có đang sử dụng PostgreSQL hay không (tự động False nếu đã fallback)."""
    if _force_sqlite:
        return False
    url = env("DATABASE_URL")
    return bool(url and (url.startswith("postgresql://") or url.startswith("postgres://")))


def _mark_postgres_failed(err: Exception) -> None:
    """Đánh dấu PostgreSQL gặp sự cố và kích hoạt fallback sang SQLite."""
    global _force_sqlite
    if (env("APP_ENV") or "").lower() == "production":
        raise ConnectionError("PostgreSQL lỗi trong production; từ chối fallback SQLite.") from err
    if not _force_sqlite:
        log.warning(
            "⚠️ Neon/PostgreSQL gặp lỗi (%s). TỰ ĐỘNG FALLBACK sang SQLite Local (%s)!",
            err,
            DB_PATH,
        )
        _force_sqlite = True
        try:
            _init_sqlite_schema()
        except Exception as e:
            log.warning("Không thể khởi tạo schema SQLite khi fallback: %s", e)


def _init_sqlite_schema_on_conn(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_key TEXT,
            topic TEXT NOT NULL,
            title TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            youtube_id TEXT,
            error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )
    try:
        cur.execute("ALTER TABLE videos ADD COLUMN run_key TEXT")
    except Exception:
        pass
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_videos_run_key ON videos(run_key) WHERE run_key IS NOT NULL")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS series (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            theme TEXT NOT NULL,
            domain TEXT,
            total_episodes INTEGER NOT NULL DEFAULT 5,
            mode TEXT NOT NULL DEFAULT 'long',
            status TEXT NOT NULL DEFAULT 'in_progress',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id TEXT PRIMARY KEY,
            run_key TEXT NOT NULL UNIQUE,
            video_id INTEGER,
            status TEXT NOT NULL,
            current_stage TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS stage_runs (
            run_id TEXT NOT NULL,
            stage TEXT NOT NULL,
            status TEXT NOT NULL,
            input_hash TEXT,
            output_json TEXT,
            error TEXT,
            started_at TEXT,
            finished_at TEXT,
            PRIMARY KEY (run_id, stage)
        )
        """
    )
    # Tự động migrate nếu bảng series cũ chưa có cột mode
    try:
        cur.execute("ALTER TABLE series ADD COLUMN mode TEXT NOT NULL DEFAULT 'long'")
    except Exception:
        pass

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS series_episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            series_id INTEGER NOT NULL REFERENCES series(id) ON DELETE CASCADE,
            episode_num INTEGER NOT NULL,
            topic TEXT NOT NULL,
            hook_from_previous TEXT,
            hook_to_next TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            video_id INTEGER REFERENCES videos(id),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(series_id, episode_num)
        );
        """
    )
    conn.commit()


def _init_sqlite_schema() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH, timeout=10.0) as conn:
        _init_sqlite_schema_on_conn(conn)


def _get_connection():
    """Tạo kết nối DB với khả năng tự động fallback Neon -> SQLite -> In-Memory."""
    global _mem_conn

    if is_postgres():
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor

            url = env("DATABASE_URL")
            conn = psycopg2.connect(url, cursor_factory=RealDictCursor, connect_timeout=8)
            conn.autocommit = True
            return conn
        except Exception as e:
            _mark_postgres_failed(e)

    # SQLite Local
    try:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        log.warning(
            "⚠️ Không thể mở SQLite file (%s: %s). FALLBACK sang SQLite In-Memory (:memory:)!",
            DB_PATH,
            e,
        )
        if _mem_conn is None:
            _mem_conn = sqlite3.connect(":memory:")
            _mem_conn.row_factory = sqlite3.Row
            _init_sqlite_schema_on_conn(_mem_conn)
        return _mem_conn


def _format_sql(sql: str) -> str:
    """Chuyển đổi placeholder '?' sang '%s' nếu đang dùng PostgreSQL."""
    if is_postgres():
        return sql.replace("?", "%s")
    return sql


def _query_all(sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        current_sql = _format_sql(sql)
        with _get_connection() as conn:
            cur = conn.cursor()
            cur.execute(current_sql, params)
            rows = cur.fetchall()
            return [dict(r) for r in rows]
    except Exception as e:
        if is_postgres():
            _mark_postgres_failed(e)
            return _query_all(sql, params)
        log.warning("Truy vấn DB thất bại (%s): %s", sql, e)
        return []


def _query_one(sql: str, params: tuple = ()) -> dict[str, Any] | None:
    try:
        current_sql = _format_sql(sql)
        with _get_connection() as conn:
            cur = conn.cursor()
            cur.execute(current_sql, params)
            row = cur.fetchone()
            return dict(row) if row else None
    except Exception as e:
        if is_postgres():
            _mark_postgres_failed(e)
            return _query_one(sql, params)
        log.warning("Truy vấn DB đơn lẻ thất bại (%s): %s", sql, e)
        return None


def _execute(sql: str, params: tuple = ()) -> None:
    try:
        current_sql = _format_sql(sql)
        with _get_connection() as conn:
            cur = conn.cursor()
            cur.execute(current_sql, params)
            if not is_postgres():
                conn.commit()
    except Exception as e:
        if is_postgres():
            _mark_postgres_failed(e)
            return _execute(sql, params)
        log.warning("Thực thi DB thất bại (%s): %s", sql, e)


def _insert_get_id(sql: str, params: tuple = ()) -> int:
    """Thực thi INSERT và trả về id vừa tạo. Có fallback an toàn nếu lỗi."""
    try:
        current_sql = _format_sql(sql)
        with _get_connection() as conn:
            cur = conn.cursor()
            if is_postgres():
                if "RETURNING" not in current_sql.upper():
                    current_sql += " RETURNING id"
                cur.execute(current_sql, params)
                row = cur.fetchone()
                return int(row["id"] if isinstance(row, dict) else row[0])
            else:
                if "RETURNING" in current_sql.upper():
                    cur.execute(current_sql, params)
                    row = cur.fetchone()
                    conn.commit()
                    return int(row[0] if row else cur.lastrowid)
                else:
                    cur.execute(current_sql, params)
                    conn.commit()
                    return int(cur.lastrowid)
    except Exception as e:
        if is_postgres():
            _mark_postgres_failed(e)
            return _insert_get_id(sql, params)
        log.warning("Insert DB thất bại (%s): %s. Dùng fallback timestamp ID.", sql, e)
        return int(datetime.utcnow().timestamp())


def init_db() -> None:
    """Khởi tạo schema cho database (tự tương thích Postgres/SQLite với fallback tự động)."""
    try:
        with _get_connection() as conn:
            cur = conn.cursor()
            if is_postgres():
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS videos (
                        id SERIAL PRIMARY KEY,
                        run_key TEXT UNIQUE,
                        topic TEXT NOT NULL,
                        title TEXT,
                        status TEXT NOT NULL DEFAULT 'pending',
                        youtube_id TEXT,
                        error TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS series (
                        id SERIAL PRIMARY KEY,
                        name TEXT NOT NULL UNIQUE,
                        theme TEXT NOT NULL,
                        domain TEXT,
                        total_episodes INTEGER NOT NULL DEFAULT 5,
                        mode TEXT NOT NULL DEFAULT 'long',
                        status TEXT NOT NULL DEFAULT 'in_progress',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS series_episodes (
                        id SERIAL PRIMARY KEY,
                        series_id INTEGER NOT NULL REFERENCES series(id) ON DELETE CASCADE,
                        episode_num INTEGER NOT NULL,
                        topic TEXT NOT NULL,
                        hook_from_previous TEXT,
                        hook_to_next TEXT,
                        status TEXT NOT NULL DEFAULT 'pending',
                        video_id INTEGER REFERENCES videos(id),
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        UNIQUE(series_id, episode_num)
                    );
                    CREATE TABLE IF NOT EXISTS pipeline_runs (
                        id TEXT PRIMARY KEY,
                        run_key TEXT NOT NULL UNIQUE,
                        video_id INTEGER,
                        status TEXT NOT NULL,
                        current_stage TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS stage_runs (
                        run_id TEXT NOT NULL,
                        stage TEXT NOT NULL,
                        status TEXT NOT NULL,
                        input_hash TEXT,
                        output_json TEXT,
                        error TEXT,
                        started_at TEXT,
                        finished_at TEXT,
                        PRIMARY KEY (run_id, stage)
                    );
                    """
                )
                cur.execute("ALTER TABLE videos ADD COLUMN IF NOT EXISTS run_key TEXT")
                cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_videos_run_key ON videos(run_key) WHERE run_key IS NOT NULL")
                try:
                    cur.execute("ALTER TABLE series ADD COLUMN IF NOT EXISTS mode TEXT NOT NULL DEFAULT 'long'")
                except Exception:
                    pass
                log.debug("Database initialized (PostgreSQL/Neon)")
            else:
                _init_sqlite_schema_on_conn(conn)
                log.debug("Database initialized (SQLite)")
    except Exception as e:
        if is_postgres():
            _mark_postgres_failed(e)
            init_db()
        else:
            log.warning("init_db thất bại: %s", e)


def _append_topic_history(topic: str) -> None:
    """Ghi topic vào file text append-only để chống trùng (luôn hoạt động độc lập với DB)."""
    try:
        TOPICS_FILE.parent.mkdir(parents=True, exist_ok=True)
        line = f"{_now_iso()}\t{topic}\n"
        with open(TOPICS_FILE, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        log.warning("Không thể ghi file %s: %s", TOPICS_FILE, e)


def record_topic_success(topic: str) -> None:
    """Ghi nhận topic đã hoàn thành vào lịch sử (chỉ gọi khi video thực sự được render thành công)."""
    _append_topic_history(topic)


def recent_topics(days: int = 30) -> list[str]:
    """Lấy danh sách các chủ đề đã dùng trong vòng N ngày."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    topics: list[str] = []

    # 1. Từ file topics_history (độc lập với DB)
    if TOPICS_FILE.exists():
        try:
            with open(TOPICS_FILE, encoding="utf-8") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    ts, _, topic = raw.partition("\t")
                    if not topic:
                        continue
                    try:
                        dt = datetime.fromisoformat(ts)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        if dt >= cutoff:
                            topics.append(topic)
                    except ValueError:
                        topics.append(topic)
        except Exception as e:
            log.debug("Đọc file topics_history lỗi: %s", e)

    # 2. Từ bảng series_episodes và videos
    try:
        rows = _query_all("SELECT topic FROM videos WHERE status != 'error' ORDER BY created_at DESC LIMIT 100")
        for r in rows:
            if r.get("topic") and r["topic"] not in topics:
                topics.append(r["topic"])
        ep_rows = _query_all("SELECT topic FROM series_episodes WHERE status != 'error' ORDER BY id DESC LIMIT 100")
        for r in ep_rows:
            if r.get("topic") and r["topic"] not in topics:
                topics.append(r["topic"])
    except Exception:
        pass

    return topics


def get_video_by_run_key(run_key: str) -> dict[str, Any] | None:
    if not run_key:
        return None
    return _query_one("SELECT * FROM videos WHERE run_key = ?", (run_key,))


def create_video(topic: str, run_key: str | None = None) -> int:
    now = _now_iso()
    try:
        if run_key:
            existing = get_video_by_run_key(run_key)
            if existing:
                return int(existing["id"])
        return _insert_get_id(
            "INSERT INTO videos (run_key, topic, status, created_at, updated_at) VALUES (?, ?, 'pending', ?, ?)",
            (run_key, topic, now, now),
        )
    except Exception as e:
        log.warning("create_video DB thất bại (%s). Dùng fallback timestamp ID.", e)
        return int(datetime.now(timezone.utc).timestamp())


def init_pipeline_run(run_id: str, run_key: str, video_id: int) -> None:
    now = _now_iso()
    _execute(
        "INSERT INTO pipeline_runs (id, run_key, video_id, status, current_stage, created_at, updated_at) "
        "VALUES (?, ?, ?, 'running', 'init', ?, ?) "
        "ON CONFLICT(run_key) DO UPDATE SET video_id = excluded.video_id, status = 'running', updated_at = excluded.updated_at",
        (run_id, run_key, video_id, now, now),
    )


def record_stage(
    run_id: str,
    stage: str,
    status: str,
    *,
    input_hash: str | None = None,
    output_json: str | None = None,
    error: str | None = None,
) -> None:
    now = _now_iso()
    started = now if status == "running" else None
    finished = now if status in {"succeeded", "failed", "skipped", "quarantined"} else None
    _execute(
        "INSERT INTO stage_runs (run_id, stage, status, input_hash, output_json, error, started_at, finished_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(run_id, stage) DO UPDATE SET status = excluded.status, input_hash = excluded.input_hash, "
        "output_json = excluded.output_json, error = excluded.error, "
        "started_at = COALESCE(stage_runs.started_at, excluded.started_at), finished_at = excluded.finished_at",
        (run_id, stage, status, input_hash, output_json, error, started, finished),
    )
    _execute(
        "UPDATE pipeline_runs SET status = ?, current_stage = ?, updated_at = ? WHERE id = ?",
        ("failed" if status == "failed" else "running", stage, now, run_id),
    )


def complete_pipeline_run(run_id: str, status: str = "completed") -> None:
    _execute(
        "UPDATE pipeline_runs SET status = ?, updated_at = ? WHERE id = ?",
        (status, _now_iso(), run_id),
    )


def update_video(video_id: int, **fields: Any) -> None:
    if not fields:
        return
    try:
        fields["updated_at"] = datetime.utcnow().isoformat()
        cols = ", ".join(f"{k} = ?" for k in fields)
        _execute(
            f"UPDATE videos SET {cols} WHERE id = ?",
            (*fields.values(), video_id),
        )
    except Exception as e:
        log.warning("update_video DB thất bại (id=%s): %s", video_id, e)


def recent_uploaded_videos_for_analytics(limit: int = 12) -> list[dict[str, Any]]:
    """Return recent uploads for delayed, optional retention collection."""
    return _query_all(
        "SELECT id, topic, title, youtube_id, created_at, updated_at FROM videos "
        "WHERE status = 'uploaded' AND youtube_id IS NOT NULL "
        "ORDER BY updated_at DESC LIMIT ?",
        (max(1, min(int(limit), 100)),),
    )


# ==============================================================================
# QUẢN LÝ CHUỖI VIDEO (SERIES & EPISODES)
# ==============================================================================


def create_series(name: str, theme: str, domain: str = "", total_episodes: int = 5, mode: str = "long") -> int:
    now = datetime.utcnow().isoformat()
    normalized_mode = "short" if mode == "short" else ("mega" if mode == "mega" else "long")
    return _insert_get_id(
        "INSERT INTO series (name, theme, domain, total_episodes, mode, status, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, 'in_progress', ?, ?)",
        (name, theme, domain, total_episodes, normalized_mode, now, now),
    )


def add_series_episode(
    series_id: int,
    episode_num: int,
    topic: str,
    hook_from_previous: str = "",
    hook_to_next: str = "",
) -> int:
    now = datetime.utcnow().isoformat()
    return _insert_get_id(
        "INSERT INTO series_episodes (series_id, episode_num, topic, hook_from_previous, hook_to_next, status, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)",
        (series_id, episode_num, topic, hook_from_previous, hook_to_next, now, now),
    )


def get_active_series(mode: str = "long") -> dict[str, Any] | None:
    """Lấy series đang chạy ('in_progress') theo đúng mode (long, short, mega) ưu tiên id cũ nhất."""
    normalized_mode = "short" if mode == "short" else ("mega" if mode == "mega" else "long")
    return _query_one(
        "SELECT * FROM series WHERE status = 'in_progress' AND (mode = ? OR (mode IS NULL AND ? = 'long')) ORDER BY id ASC LIMIT 1",
        (normalized_mode, normalized_mode),
    )


def get_series_by_id(series_id: int) -> dict[str, Any] | None:
    return _query_one("SELECT * FROM series WHERE id = ?", (series_id,))


def get_series_episodes(series_id: int) -> list[dict[str, Any]]:
    return _query_all("SELECT * FROM series_episodes WHERE series_id = ? ORDER BY episode_num ASC", (series_id,))


def get_next_pending_episode(series_id: int) -> dict[str, Any] | None:
    """Lấy tập pending/lỗi/quarantine cần thử lại tiếp theo của series."""
    return _query_one(
        "SELECT * FROM series_episodes WHERE series_id = ? AND status IN ('pending', 'error', 'quarantined') ORDER BY episode_num ASC LIMIT 1",
        (series_id,),
    )


def update_series(series_id: int, **fields: Any) -> None:
    if not fields:
        return
    try:
        fields["updated_at"] = datetime.utcnow().isoformat()
        cols = ", ".join(f"{k} = ?" for k in fields)
        _execute(f"UPDATE series SET {cols} WHERE id = ?", (*fields.values(), series_id))
    except Exception as e:
        log.warning("update_series DB thất bại (id=%s): %s", series_id, e)


def update_episode(episode_id: int, **fields: Any) -> None:
    if not fields:
        return
    try:
        fields["updated_at"] = datetime.utcnow().isoformat()
        cols = ", ".join(f"{k} = ?" for k in fields)
        _execute(f"UPDATE series_episodes SET {cols} WHERE id = ?", (*fields.values(), episode_id))
    except Exception as e:
        log.warning("update_episode DB thất bại (id=%s): %s", episode_id, e)


def list_all_series(mode: str | None = None) -> list[dict[str, Any]]:
    """Liệt kê toàn bộ các series cùng thống kê số tập đã hoàn thành, có thể lọc theo mode."""
    if mode:
        normalized_mode = "short" if mode == "short" else ("mega" if mode == "mega" else "long")
        series_list = _query_all(
            "SELECT * FROM series WHERE mode = ? OR (mode IS NULL AND ? = 'long') ORDER BY id DESC",
            (normalized_mode, normalized_mode),
        )
    else:
        series_list = _query_all("SELECT * FROM series ORDER BY id DESC")
    for s in series_list:
        eps = get_series_episodes(s["id"])
        completed = sum(1 for e in eps if e.get("status") in ("rendered", "uploaded"))
        s["episodes"] = eps
        s["completed_episodes"] = completed
    return series_list
