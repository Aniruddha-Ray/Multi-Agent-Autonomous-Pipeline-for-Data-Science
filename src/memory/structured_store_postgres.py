"""PostgreSQL-backed structured memory of past pipeline runs.

Stage G: drop-in backend replacement for StructuredMemory
(memory/structured_store.py, SQLite). Implements the exact same public
method set — __init__, run_startup_migrations, add_run, get_run, all_runs,
count — so MemoryRepository (memory/repository.py) works identically
regardless of which one it's constructed with.

The one unavoidable signature difference: __init__ takes a Postgres DSN
(connection string) instead of a SQLite file path — there is no
file-path equivalent for a network database, so this is a deliberate,
minimal deviation, not an oversight.

CRITICAL COMPATIBILITY DETAIL (see Stage G analysis, "Issue 1"): psycopg2
auto-parses JSONB columns into Python dicts on SELECT by default.
MemoryRepository's existing code (unchanged, Stage E4) calls
json.loads(run["metrics_json"]) etc., expecting a STRING. To avoid forcing
any change in memory/repository.py, every *_json column is explicitly
cast to ::text in get_run/all_runs' SELECT clauses below, so this class
returns exactly the same shape (dict[str, Any] with *_json fields as
plain strings) that StructuredMemory already returns. The columns
themselves are still stored as real JSONB (queryable/indexable in
Postgres) — only the read-path shape is normalized to match SQLite.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional

import psycopg2
from psycopg2.extras import Json

# Every *_json column, cast to ::text on read so psycopg2 returns a plain
# string (matching SQLite's TEXT-column behavior) instead of an
# auto-parsed dict. Keeping this as one explicit column list (rather than
# SELECT *) is what makes the cast possible.
_ALL_COLUMNS_TEXT_CAST_SELECT = """
    run_id, created_at,
    dataset_summary_json::text AS dataset_summary_json,
    planner_reasoning_json::text AS planner_reasoning_json,
    chosen_model,
    metrics_json::text AS metrics_json,
    critic_notes_json::text AS critic_notes_json,
    embedding_json::text AS embedding_json,
    experience_payload_json::text AS experience_payload_json,
    memory_quality, experience_score, confidence, success_rate,
    retrieval_count, last_retrieved, last_updated, deleted_at
"""


def _json_or_none(value: Any) -> Optional[Json]:
    if value is None:
        return None
    return Json(value, dumps=lambda o: json.dumps(o, default=str))


class PostgresStructuredMemory:
    """PostgreSQL-backed structured memory of past pipeline runs.

    Same public API as StructuredMemory (SQLite) — see that class's
    docstring for what each method does; behavior described there applies
    identically here except where explicitly noted (JSONB storage,
    ::text-cast reads).
    """

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self.conn = psycopg2.connect(dsn)
        self.conn.autocommit = False
        self._create_schema()

    def _create_schema(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id SERIAL PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    dataset_summary_json JSONB NOT NULL,
                    planner_reasoning_json JSONB,
                    chosen_model TEXT,
                    metrics_json JSONB,
                    critic_notes_json JSONB,
                    embedding_json JSONB NOT NULL,
                    experience_payload_json JSONB
                )
                """
            )
        self.conn.commit()

    def run_startup_migrations(self) -> None:
        """Same role as StructuredMemory.run_startup_migrations() (Stage G)
        — additive ALTER TABLE guards, safe to call every startup."""
        self._ensure_experience_payload_column()
        self._ensure_memory_quality_columns()
        self._ensure_soft_delete_column()

    def _column_exists(self, column: str) -> bool:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = %s AND column_name = %s",
                ("runs", column),
            )
            return cur.fetchone() is not None

    def _ensure_experience_payload_column(self) -> None:
        if not self._column_exists("experience_payload_json"):
            with self.conn.cursor() as cur:
                cur.execute("ALTER TABLE runs ADD COLUMN experience_payload_json JSONB")
            self.conn.commit()

    def _ensure_memory_quality_columns(self) -> None:
        new_cols = {
            "memory_quality": "REAL", "experience_score": "REAL", "confidence": "REAL",
            "success_rate": "REAL", "retrieval_count": "INTEGER DEFAULT 0",
            "last_retrieved": "TEXT", "last_updated": "TEXT",
        }
        with self.conn.cursor() as cur:
            for col, col_type in new_cols.items():
                if not self._column_exists(col):
                    cur.execute(f"ALTER TABLE runs ADD COLUMN {col} {col_type}")
        self.conn.commit()

    def _ensure_soft_delete_column(self) -> None:
        if not self._column_exists("deleted_at"):
            with self.conn.cursor() as cur:
                cur.execute("ALTER TABLE runs ADD COLUMN deleted_at TEXT")
            self.conn.commit()

    def add_run(
        self,
        dataset_summary: dict[str, Any],
        embedding,
        planner_reasoning: Optional[dict[str, Any]] = None,
        chosen_model: Optional[str] = None,
        metrics: Optional[dict[str, Any]] = None,
        critic_notes: Optional[dict[str, Any]] = None,
        experience_payload: Optional[dict[str, Any]] = None,
    ) -> int:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO runs (
                    created_at, dataset_summary_json, planner_reasoning_json,
                    chosen_model, metrics_json, critic_notes_json, embedding_json,
                    experience_payload_json
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING run_id
                """,
                (
                    datetime.utcnow().isoformat(),
                    _json_or_none(dataset_summary),
                    _json_or_none(planner_reasoning),
                    chosen_model,
                    _json_or_none(metrics),
                    _json_or_none(critic_notes),
                    Json(embedding.tolist()),
                    _json_or_none(experience_payload),
                ),
            )
            run_id = cur.fetchone()[0]
        self.conn.commit()
        return int(run_id)

    def get_run(self, run_id: int) -> Optional[dict[str, Any]]:
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT {_ALL_COLUMNS_TEXT_CAST_SELECT} FROM runs WHERE run_id = %s", (run_id,))
            row = cur.fetchone()
            if row is None:
                return None
            cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))

    def all_runs(self) -> list[dict[str, Any]]:
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT {_ALL_COLUMNS_TEXT_CAST_SELECT} FROM runs")
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
        return [dict(zip(cols, row)) for row in rows]

    def count(self) -> int:
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM runs")
            return int(cur.fetchone()[0])