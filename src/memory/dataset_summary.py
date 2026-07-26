"""Dataset summary helpers for memory storage and retrieval.

Extracted from Notebook Cell 17 ("CELL 6h — MEMORY HELPERS")
— ``build_dataset_summary`` and ``retrieve_similar_runs`` — and Notebook
Cell 18 ("NEW CELL — MEMORY RETRIEVAL: DATASET SUMMARY GENERATOR")
— ``_compute_outlier_summary`` only.

``retrieve_similar_runs`` originally closed over the notebook's module-level
``semantic_memory``/``structured_memory`` globals. Per the Stage E4 Issue A
decision, it now takes ``structured``/``semantic`` as explicit parameters —
a pure software-engineering refactor, matching the dependency-injection style
already used by ``MemoryRepository`` (Stage E4). The FAISS search, SQLite
hydration, returned dict shape, and the malformed-record skip-and-log
behavior are all unchanged from the notebook.

NOT included in this module (Stage E4 Issue B decision — deferred to
Stage E5): ``build_semantic_dataset_summary`` (Cell 18), since it depends on
``_top_correlated_pairs``, which stays in the EDA agent module
(``agents/eda.py``, Cell 6) and is not duplicated or relocated here.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

import sys
import os
from pathlib import Path 

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.memory.semantic_store import SemanticMemory
from src.memory.structured_store import StructuredMemory


def build_dataset_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    """Compact dataset fingerprint used both for the SQLite record and as the
    input to ``embed_dataset_summary`` (``memory.semantic_store``)."""
    return {
        "n_rows": metadata["n_rows"],
        "n_cols": metadata["n_cols"],
        "pct_missing": metadata["pct_missing"],
        "imbalance_ratio": metadata["imbalance_ratio"],
        "n_categorical": metadata["n_categorical"],
        "n_numerical": metadata["n_numerical"],
        "avg_cardinality": metadata["avg_cardinality"],
        "problem_type": metadata["problem_type"],
    }


def retrieve_similar_runs(
    structured: StructuredMemory,
    semantic: SemanticMemory,
    embedding: np.ndarray,
    k: int = 3,
) -> list[dict[str, Any]]:
    """Look up the k nearest past runs in the FAISS index and hydrate them
    from SQLite. Skips and logs any malformed record instead of letting one
    bad row abort retrieval for the rest (Task 6, architecture audit).

    ``structured``/``semantic`` are passed explicitly (Stage E4, Issue A)
    rather than read from module-level globals, so this function has no
    hidden dependency on process-wide singleton state — callers (the
    Planner's fallback path, Stage E5) must supply the same
    ``StructuredMemory``/``SemanticMemory`` instances the rest of the
    pipeline uses.
    """
    hits = semantic.search(embedding, k=k)
    results: list[dict[str, Any]] = []
    for run_id, distance in hits:
        try:
            run = structured.get_run(run_id)
            if run is None:
                continue
            results.append({
                "run_id": run_id,
                "distance": distance,
                "chosen_model": run.get("chosen_model"),
                "metrics": json.loads(run["metrics_json"]) if run.get("metrics_json") else None,
                "created_at": run.get("created_at"),
            })
        except (json.JSONDecodeError, ValueError, TypeError, KeyError) as exc:
            print(f"[retrieve_similar_runs] Skipping malformed memory run_id={run_id}: {exc}")
            continue
    return results


def _compute_outlier_summary(df: pd.DataFrame, numerical_columns: list[str]) -> dict[str, int]:
    """IQR-based outlier count per numerical column. Deterministic, no LLM."""
    summary: dict[str, int] = {}
    for col in numerical_columns:
        series = df[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            summary[col] = 0
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        summary[col] = int(((series < lower) | (series > upper)).sum())
    return summary

# --- appended this stage: resolves Stage E4 Issue B -------------------------
from src.agents.eda import _top_correlated_pairs  # noqa: E402  (Issue B resolution — kept in EDA domain, not duplicated)


def build_semantic_dataset_summary(metadata: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    """Deterministic, compact dataset fingerprint for memory retrieval.

    Reuses ``build_dataset_summary`` (this module) for the fields it already
    covers, then adds target info, correlation characteristics, and an
    outlier summary. Never includes raw rows/values — only aggregates.
    """
    base = build_dataset_summary(metadata)

    corr_df = pd.DataFrame(metadata["correlation_matrix"])
    top_pairs = _top_correlated_pairs(metadata, top_k=3)

    base.update({
        "target_column": metadata["target_column"],
        "class_counts": metadata.get("class_counts", {}),
        "high_cardinality_columns": metadata["high_cardinality_columns"],
        "correlation_characteristics": {
            "n_numerical_pairs_checked": max(0, corr_df.shape[1] * (corr_df.shape[1] - 1) // 2) if not corr_df.empty else 0,
            "top_pairs": [{"a": a, "b": b, "r": round(v, 3)} for a, b, v in top_pairs],
        },
        "outlier_summary": _compute_outlier_summary(df, metadata["numerical_columns"]),
    })
    return base