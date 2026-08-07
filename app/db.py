"""
Database layer – PostgreSQL storage for task records.
Uses psycopg2 for production; falls back to SQLite for local dev / testing.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from dataclasses import dataclass


@dataclass
class TaskRecord:
    task_id: str
    task_type: str
    status: str  # pending | processing | completed | failed | cancelled
    payload: dict
    priority: int = 5
    callback_url: str | None = None
    created_at: str = ""
    started_at: str | None = None
    completed_at: str | None = None
    result: dict | None = None
    error: str | None = None
    processing_time_ms: float | None = None
    worker_id: str | None = None


# ---------------------------------------------------------------------------
# SQLite-backed store (local / test)
# ---------------------------------------------------------------------------

class SQLiteDB:
    """Thread-safe SQLite storage. Swap for PostgreSQL via DATABASE_URL."""

    def __init__(self, db_path: str = "tasks.db"):
        self.db_path = db_path
        self._local = threading.local()

    @property
    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def init(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                payload TEXT NOT NULL,
                priority INTEGER DEFAULT 5,
                callback_url TEXT,
                created_at TEXT,
                started_at TEXT,
                completed_at TEXT,
                result TEXT,
                error TEXT,
                processing_time_ms REAL,
                worker_id TEXT
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON tasks(status)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_type ON tasks(task_type)")
        self._conn.commit()

    def insert_task(self, t: TaskRecord):
        self._conn.execute(
            """INSERT INTO tasks
               (task_id, task_type, status, payload, priority, callback_url, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (t.task_id, t.task_type, t.status, json.dumps(t.payload),
             t.priority, t.callback_url, t.created_at),
        )
        self._conn.commit()

    def get_task(self, task_id: str) -> TaskRecord | None:
        row = self._conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        return self._row_to_record(row) if row else None

    def update_task(self, task_id: str, **kwargs):
        sets = []
        vals = []
        for k, v in kwargs.items():
            if k in ("result", "payload") and isinstance(v, dict):
                v = json.dumps(v)
            sets.append(f"{k} = ?")
            vals.append(v)
        vals.append(task_id)
        self._conn.execute(f"UPDATE tasks SET {', '.join(sets)} WHERE task_id = ?", vals)
        self._conn.commit()

    def list_tasks(self, status: str | None = None, task_type: str | None = None,
                   limit: int = 50, offset: int = 0) -> list[TaskRecord]:
        q = "SELECT * FROM tasks WHERE 1=1"
        params: list = []
        if status:
            q += " AND status = ?"
            params.append(status)
        if task_type:
            q += " AND task_type = ?"
            params.append(task_type)
        q += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params += [limit, offset]
        rows = self._conn.execute(q, params).fetchall()
        return [self._row_to_record(r) for r in rows]

    def count_tasks(self, status: str | None = None, task_type: str | None = None) -> int:
        q = "SELECT COUNT(*) FROM tasks WHERE 1=1"
        params: list = []
        if status:
            q += " AND status = ?"
            params.append(status)
        if task_type:
            q += " AND task_type = ?"
            params.append(task_type)
        return self._conn.execute(q, params).fetchone()[0]

    def avg_processing_time(self) -> float:
        row = self._conn.execute(
            "SELECT AVG(processing_time_ms) FROM tasks WHERE status = 'completed'"
        ).fetchone()
        return row[0] or 0.0

    def _row_to_record(self, row) -> TaskRecord:
        return TaskRecord(
            task_id=row["task_id"],
            task_type=row["task_type"],
            status=row["status"],
            payload=json.loads(row["payload"]) if row["payload"] else {},
            priority=row["priority"],
            callback_url=row["callback_url"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            result=json.loads(row["result"]) if row["result"] else None,
            error=row["error"],
            processing_time_ms=row["processing_time_ms"],
            worker_id=row["worker_id"],
        )


# ---------------------------------------------------------------------------
# PostgreSQL-backed store (production)
# ---------------------------------------------------------------------------

class PostgresDB:
    """Production PostgreSQL storage."""

    def __init__(self, dsn: str):
        import psycopg2
        import psycopg2.extras
        self.dsn = dsn
        self._conn = psycopg2.connect(dsn)
        self._conn.autocommit = True

    def init(self):
        with self._conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    task_type TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    payload JSONB NOT NULL,
                    priority INTEGER DEFAULT 5,
                    callback_url TEXT,
                    created_at TIMESTAMPTZ,
                    started_at TIMESTAMPTZ,
                    completed_at TIMESTAMPTZ,
                    result JSONB,
                    error TEXT,
                    processing_time_ms DOUBLE PRECISION,
                    worker_id TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
                CREATE INDEX IF NOT EXISTS idx_tasks_type ON tasks(task_type);
                CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_at DESC);
            """)

    def insert_task(self, t: TaskRecord):
        with self._conn.cursor() as cur:
            cur.execute(
                """INSERT INTO tasks
                   (task_id, task_type, status, payload, priority, callback_url, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (t.task_id, t.task_type, t.status, json.dumps(t.payload),
                 t.priority, t.callback_url, t.created_at),
            )

    def get_task(self, task_id: str) -> TaskRecord | None:
        with self._conn.cursor() as cur:
            cur.execute("SELECT * FROM tasks WHERE task_id = %s", (task_id,))
            row = cur.fetchone()
            if row is None:
                return None
            cols = [d[0] for d in cur.description]
            d = dict(zip(cols, row))
            return TaskRecord(**{k: d[k] for k in TaskRecord.__dataclass_fields__})

    def update_task(self, task_id: str, **kwargs):
        sets = []
        vals = []
        for k, v in kwargs.items():
            if k in ("result", "payload") and isinstance(v, dict):
                v = json.dumps(v)
            sets.append(f"{k} = %s")
            vals.append(v)
        vals.append(task_id)
        with self._conn.cursor() as cur:
            cur.execute(f"UPDATE tasks SET {', '.join(sets)} WHERE task_id = %s", vals)

    def list_tasks(self, status=None, task_type=None, limit=50, offset=0):
        q = "SELECT * FROM tasks WHERE 1=1"
        params: list = []
        if status:
            q += " AND status = %s"
            params.append(status)
        if task_type:
            q += " AND task_type = %s"
            params.append(task_type)
        q += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
        params += [limit, offset]
        with self._conn.cursor() as cur:
            cur.execute(q, params)
            cols = [d[0] for d in cur.description]
            return [TaskRecord(**dict(zip(cols, row))) for row in cur.fetchall()]

    def count_tasks(self, status=None, task_type=None) -> int:
        q = "SELECT COUNT(*) FROM tasks WHERE 1=1"
        params: list = []
        if status:
            q += " AND status = %s"
            params.append(status)
        if task_type:
            q += " AND task_type = %s"
            params.append(task_type)
        with self._conn.cursor() as cur:
            cur.execute(q, params)
            return cur.fetchone()[0]

    def avg_processing_time(self) -> float:
        with self._conn.cursor() as cur:
            cur.execute("SELECT AVG(processing_time_ms) FROM tasks WHERE status = 'completed'")
            r = cur.fetchone()[0]
            return r or 0.0


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_db = None


def get_db():
    global _db
    if _db is None:
        dsn = os.getenv("DATABASE_URL")
        if dsn:
            _db = PostgresDB(dsn)
        else:
            _db = SQLiteDB(os.getenv("SQLITE_PATH", "tasks.db"))
    return _db


def init_db():
    get_db().init()
