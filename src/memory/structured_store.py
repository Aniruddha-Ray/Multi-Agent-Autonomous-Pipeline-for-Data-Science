"""SQLite-backed structured memory of past pipeline runs.

Extracted verbatim from Notebook Cell 3 ("CELL 4 — MEMORY INITIALIZATION")
— the ``StructuredMemory`` class and its ``_ensure_experience_payload_column``
migration guard — plus Notebook Cell 15
("NEW CELL — MEMORY QUALITY: SCHEMA MIGRATION")'s
``_ensure_memory_quality_columns``, merged into this same file per the
Cell-to-Module Mapping decision: both are additive migrations on the same
``runs`` table this class owns, and keeping migrations physically separate
from the schema they migrate would be a source of drift.

NOT included in this module: the notebook's own instantiation
(``structured_memory = StructuredMemory(CFG.sqlite_path)``) and the two
migration-guard *calls* against that instance. Those are side effects
deferred to the Stage E8 composition root (``main.py``).
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any, Optional


class StructuredMemory:
    """SQLite-backed structured memory of past pipeline runs."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self._create_schema()

    def _create_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                dataset_summary_json TEXT NOT NULL,
                planner_reasoning_json TEXT,
                chosen_model TEXT,
                metrics_json TEXT,
                critic_notes_json TEXT,
                embedding_json TEXT NOT NULL,
                experience_payload_json TEXT
            )
            """
        )
        self.conn.commit()

    def add_run(self, dataset_summary, embedding, planner_reasoning=None,
                chosen_model=None, metrics=None, critic_notes=None,
                experience_payload=None) -> int:
        """``experience_payload`` (Implementation Task 1/2, persistence audit):
        the full versioned Experience payload (schema_version, planner
        decision, executed preprocessing, transformation summary, training
        configuration, evaluation metrics, explainability summary, critic
        observations, experience score) — stored as a single JSON blob
        rather than one column per field, so PostgreSQL migration only
        needs to swap this column's type to JSONB. Existing searchable
        columns (dataset_summary_json, chosen_model, metrics_json,
        critic_notes_json, embedding_json, created_at) are unchanged and
        keep working exactly as before this payload existed.
        """
        cur = self.conn.execute(
            """
            INSERT INTO runs (
                created_at, dataset_summary_json, planner_reasoning_json,
                chosen_model, metrics_json, critic_notes_json, embedding_json,
                experience_payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.utcnow().isoformat(),
                json.dumps(dataset_summary, default=str),
                json.dumps(planner_reasoning, default=str) if planner_reasoning else None,
                chosen_model,
                json.dumps(metrics, default=str) if metrics else None,
                json.dumps(critic_notes, default=str) if critic_notes else None,
                json.dumps(embedding.tolist()),
                json.dumps(experience_payload, default=str) if experience_payload else None,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def get_run(self, run_id: int) -> Optional[dict[str, Any]]:
        cur = self.conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        row = cur.fetchone()
        if row is None:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))

    def all_runs(self) -> list[dict[str, Any]]:
        cur = self.conn.execute("SELECT * FROM runs")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def count(self) -> int:
        cur = self.conn.execute("SELECT COUNT(*) FROM runs")
        return int(cur.fetchone()[0])


def _ensure_experience_payload_column(conn: sqlite3.Connection) -> None:
    """Migration guard (Implementation Task 1/6, persistence audit): a
    ``runs`` table created by an older notebook run (before this column
    existed) won't get ``experience_payload_json`` from
    ``CREATE TABLE IF NOT EXISTS`` alone, since that only applies to brand
    new tables. Adds the column if it's missing, exactly like
    ``_ensure_memory_quality_columns`` below. Safe to call every startup —
    a no-op once the column exists.
    """
    cols = [row[1] for row in conn.execute("PRAGMA table_info(runs)").fetchall()]
    if "experience_payload_json" not in cols:
        conn.execute("ALTER TABLE runs ADD COLUMN experience_payload_json TEXT")
        conn.commit()


def _ensure_memory_quality_columns(conn: sqlite3.Connection) -> None:
    """Extracted verbatim from Notebook Cell 15
    ("NEW CELL — MEMORY QUALITY: SCHEMA MIGRATION")."""
    cols = [row[1] for row in conn.execute("PRAGMA table_info(runs)").fetchall()]
    new_cols = {
        "memory_quality": "REAL", "experience_score": "REAL", "confidence": "REAL",
        "success_rate": "REAL", "retrieval_count": "INTEGER DEFAULT 0",
        "last_retrieved": "TEXT", "last_updated": "TEXT",
    }
    for col, col_type in new_cols.items():
        if col not in cols:
            conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {col_type}")
    conn.commit()