"""Reusable façade over StructuredMemory + SemanticMemory + EmbeddingProvider.

Extracted verbatim from Notebook Cell 20
("NEW CELL — MEMORY RETRIEVAL: MEMORY REPOSITORY").

NOT included in this module: ``memory_repository = MemoryRepository(...)`` —
that instantiation is deferred to the Stage E8 composition root.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional

import numpy as np
import sys
import os
from pathlib import Path 

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.memory.embeddings import EmbeddingProvider
from src.memory.semantic_store import SemanticMemory
from src.memory.structured_store import StructuredMemory


# def _ensure_soft_delete_column(conn) -> None:
#     """Additive schema migration — safe to run on an existing DB file."""
#     cols = [row[1] for row in conn.execute("PRAGMA table_info(runs)").fetchall()]
#     if "deleted_at" not in cols:
#         conn.execute("ALTER TABLE runs ADD COLUMN deleted_at TEXT")
#         conn.commit()


LEGACY_EXPERIENCE_SCHEMA_VERSION: int = 1  # Implementation Task 6, persistence audit


def _normalize_experience_payload(raw_payload: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Implementation Task 6 (persistence audit): make every retrieved
    memory's experience_payload safe to read regardless of which schema
    version — or none at all — wrote it.

    - A run persisted before ``experience_payload_json`` existed (or one
      whose payload failed to parse upstream) has ``raw_payload is None`` —
      this is a legacy record, not an error. It gets a stub dict with
      ``schema_version=LEGACY_EXPERIENCE_SCHEMA_VERSION`` and every detail
      field set to ``None``, so downstream ``.get(...)`` calls never
      KeyError/AttributeError on it.
    - A parsed payload missing its own ``schema_version`` key (shouldn't
      happen once every write goes through ``build_experience_payload``,
      but handled defensively) is treated the same way, without discarding
      whatever fields it does have.

    Callers should never assume every stored experience uses the latest
    schema — always read through this normalizer rather than the raw
    column.
    """
    if not raw_payload:
        return {
            "schema_version": LEGACY_EXPERIENCE_SCHEMA_VERSION,
            "planner_decision": None,
            "executed_preprocessing": None,
            "transformation_summary": None,
            "training_configuration": None,
            "evaluation_metrics": None,
            "explainability_summary": None,
            "critic_observations": None,
            "experience_score": None,
        }
    return {
        "schema_version": raw_payload.get("schema_version", LEGACY_EXPERIENCE_SCHEMA_VERSION),
        "planner_decision": raw_payload.get("planner_decision"),
        "executed_preprocessing": raw_payload.get("executed_preprocessing"),
        "transformation_summary": raw_payload.get("transformation_summary"),
        "training_configuration": raw_payload.get("training_configuration"),
        "evaluation_metrics": raw_payload.get("evaluation_metrics"),
        "explainability_summary": raw_payload.get("explainability_summary"),
        "critic_observations": raw_payload.get("critic_observations"),
        "experience_score": raw_payload.get("experience_score"),
    }


class MemoryRepository:
    """Reusable façade over the existing StructuredMemory + SemanticMemory +
    EmbeddingProvider. Does not change any of their internals — this is
    intentionally additive so a future Postgres/Qdrant swap only needs new
    implementations of the objects passed into __init__.
    """

    # def __init__(
    #     self,
    #     structured: StructuredMemory,
    #     semantic: SemanticMemory,
    #     embedding_provider: EmbeddingProvider,
    # ) -> None:
    #     self.structured = structured
    #     self.semantic = semantic
    #     self.embedding_provider = embedding_provider
    #     _ensure_soft_delete_column(self.structured.conn)
    #     self._rehydrate_semantic_index()

    def __init__(
            self,
            structured: Any,  # StructuredMemory (SQLite) or PostgresStructuredMemory (Stage G) — both
                            # satisfy the same duck-typed interface: run_startup_migrations(),
                            # add_run(), get_run(), all_runs(), count(), .conn
            semantic: SemanticMemory,
            embedding_provider: EmbeddingProvider,
        ) -> None:
            self.structured = structured
            self.semantic = semantic
            self.embedding_provider = embedding_provider
            self.structured.run_startup_migrations()
            self._rehydrate_semantic_index()


    def _rehydrate_semantic_index(self) -> None:
        """Rebuild the in-process FAISS index from SQLite on startup.

        SemanticMemory's IndexFlatL2 lives only in process memory — every
        kernel restart creates a fresh, empty index even though
        StructuredMemory's SQLite file still has every past run. Without
        this, retrieval silently finds nothing after any "Restart & Run
        All," which looks exactly like "memory retrieval isn't working"
        even though the runs are all still on disk. Only runs on a fresh
        index (ntotal == 0) with existing SQLite rows, so it never
        double-adds vectors within a session that already built its index.
        """
        if self.semantic.index.ntotal > 0:
            return
        runs = [r for r in self.structured.all_runs() if not r.get("deleted_at")]
        if not runs:
            return
        n_skipped = 0
        for run in runs:
            try:
                embedding = np.array(json.loads(run["embedding_json"]), dtype="float32")
                if embedding.ndim != 1 or embedding.shape[0] != self.semantic.dim:
                    raise ValueError(
                        f"embedding has shape {embedding.shape}, expected ({self.semantic.dim},)"
                    )
            except Exception as exc:  # noqa: BLE001 — malformed row must not abort rehydration
                n_skipped += 1
                print(f"[MemoryRepository] Skipping malformed memory run_id={run.get('run_id')} "
                      f"during rehydration: {exc}")
                continue
            self.semantic.add(run["run_id"], embedding)
        n_loaded = len(runs) - n_skipped
        print(f"MemoryRepository: rehydrated FAISS index with {n_loaded} run(s) from SQLite "
              f"(existing memory now searchable after this restart)"
              + (f", skipped {n_skipped} malformed record(s)." if n_skipped else "."))

    # ---- write path -----------------------------------------------------
    def save_memory(
        self,
        dataset_summary: dict[str, Any],
        planner_reasoning: Optional[dict[str, Any]] = None,
        chosen_model: Optional[str] = None,
        metrics: Optional[dict[str, Any]] = None,
        critic_notes: Optional[dict[str, Any]] = None,
        experience_payload: Optional[dict[str, Any]] = None,
    ) -> int:
        """``experience_payload`` (Implementation Task 1/2, persistence audit):
        the full versioned Experience payload — optional so every existing
        caller keeps working unchanged; passed straight through to
        ``StructuredMemory.add_run`` for JSON storage alongside the existing
        searchable columns (Implementation Task 3 — nothing below is removed
        or replaced, only complemented).
        """
        embedding = self.embedding_provider.embed(dataset_summary)
        run_id = self.structured.add_run(
            dataset_summary, embedding, planner_reasoning, chosen_model, metrics, critic_notes,
            experience_payload=experience_payload,
        )
        self.semantic.add(run_id, embedding)
        return run_id

    def update_memory(self, run_id: int, **fields: Any) -> bool:
        """Update mutable columns (chosen_model, metrics_json, critic_notes_json)
        on an existing run. Does not touch the embedding or FAISS index."""
        allowed = {
            "chosen_model", "metrics_json", "critic_notes_json", "planner_reasoning_json",
            "experience_payload_json",  # NEW — Implementation Task 1/2, persistence audit
        }
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return False
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = [json.dumps(v, default=str) if k.endswith("_json") and not isinstance(v, str) else v
                  for k, v in updates.items()]
        cur = self.structured.conn.execute(
            f"UPDATE runs SET {set_clause} WHERE run_id = ?", (*values, run_id)
        )
        self.structured.conn.commit()
        return cur.rowcount > 0

    def delete_memory(self, run_id: int) -> bool:
        """Soft delete: marks the row so it's excluded from retrieval/listing.
        FAISS IndexFlatL2 has no per-vector remove, so the vector stays in
        the index but is filtered out post-search — safe, no index rebuild."""
        cur = self.structured.conn.execute(
            "UPDATE runs SET deleted_at = ? WHERE run_id = ? AND deleted_at IS NULL",
            (datetime.utcnow().isoformat(), run_id),
        )
        self.structured.conn.commit()
        return cur.rowcount > 0

    def set_quality(self, run_id: int, **quality_fields: Any) -> None:
        """Writes memory_quality/experience_score/confidence/success_rate
        (columns added by the Memory Quality schema migration cell)."""
        allowed = {"memory_quality", "experience_score", "confidence", "success_rate"}
        updates = {k: v for k, v in quality_fields.items() if k in allowed}
        if not updates:
            return
        updates["last_updated"] = datetime.utcnow().isoformat()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        self.structured.conn.execute(
            f"UPDATE runs SET {set_clause} WHERE run_id = ?", (*updates.values(), run_id)
        )
        self.structured.conn.commit()

    def _touch_retrieved(self, run_ids: list[int]) -> None:
        """Bumps retrieval_count / last_retrieved for every memory that was
        actually returned by retrieve_memories()."""
        if not run_ids:
            return
        now = datetime.utcnow().isoformat()
        for run_id in run_ids:
            self.structured.conn.execute(
                "UPDATE runs SET retrieval_count = COALESCE(retrieval_count, 0) + 1, "
                "last_retrieved = ? WHERE run_id = ?", (now, run_id),
            )
        self.structured.conn.commit()

    # ---- read path --------------------------------------------------------
    def list_memories(self, limit: Optional[int] = None) -> list[dict[str, Any]]:
        runs = [r for r in self.structured.all_runs() if not r.get("deleted_at")]
        runs.sort(key=lambda r: r["created_at"], reverse=True)
        return runs[:limit] if limit else runs

    def retrieve_memories(
        self,
        dataset_summary: dict[str, Any],
        k: int = 5,
        min_similarity: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Top-K semantic search with true cosine similarity.

        FAISS holds an IndexFlatL2 (unchanged). ``embed_dataset_summary``
        L2-normalizes every embedding at the source, so both the stored
        vectors and this query vector are unit length — for unit vectors,
        L2 distance and cosine similarity are monotonically related
        (cos = 1 - d²/2), giving an exact cosine score without touching
        SemanticMemory's index type. The normalization here is a cheap,
        idempotent safety net in case a caller ever supplies a
        non-unit-length embedding directly. Over-fetches 3x candidates to
        survive the deleted-row filter and the similarity threshold before
        truncating to k.
        """
        raw_embedding = self.embedding_provider.embed(dataset_summary)
        norm = np.linalg.norm(raw_embedding)
        query_vec = raw_embedding / norm if norm > 0 else raw_embedding

        hits = self.semantic.search(query_vec, k=max(k * 3, k))
        results: list[dict[str, Any]] = []
        for run_id, l2_sq_distance in hits:
            try:
                run = self.structured.get_run(run_id)
                if run is None or run.get("deleted_at"):
                    continue
                if "dataset_summary_json" not in run or "created_at" not in run:
                    raise ValueError("memory record is missing required field(s)")

                similarity = max(0.0, min(1.0, 1.0 - l2_sq_distance / 2.0))
                if similarity < min_similarity:
                    continue

                metrics = json.loads(run["metrics_json"]) if run.get("metrics_json") else None
                critic_notes = json.loads(run["critic_notes_json"]) if run.get("critic_notes_json") else None
                planner_reasoning = (
                    json.loads(run["planner_reasoning_json"]) if run.get("planner_reasoning_json") else None
                )
                raw_experience_payload = (
                    json.loads(run["experience_payload_json"]) if run.get("experience_payload_json") else None
                )
                experience_payload = _normalize_experience_payload(raw_experience_payload)
            except (json.JSONDecodeError, ValueError, TypeError, KeyError) as exc:
                print(f"[MemoryRepository] Skipping malformed memory run_id={run_id} "
                      f"during retrieval: {exc}")
                continue

            results.append({
                "run_id": run_id,
                "similarity": round(similarity, 4),
                "chosen_model": run.get("chosen_model"),
                "metrics": metrics,
                "critic_notes": critic_notes,
                "planner_reasoning": planner_reasoning,
                "created_at": run.get("created_at"),
                "memory_quality": run.get("memory_quality"),
                "experience_payload": experience_payload,
            })
            if len(results) >= k:
                break
        self._touch_retrieved([r["run_id"] for r in results])
        return results