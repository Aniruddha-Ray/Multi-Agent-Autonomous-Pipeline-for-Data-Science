"""FAISS-backed semantic index over dataset-summary embeddings.

Extracted verbatim from Notebook Cell 3 ("CELL 4 — MEMORY INITIALIZATION")
— the ``SemanticMemory`` class and ``embed_dataset_summary`` function only.
"""
from __future__ import annotations

from typing import Any

import faiss
import numpy as np
import hashlib


class SemanticMemory:
    """FAISS-backed semantic index over dataset-summary embeddings."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.run_ids: list[int] = []

    def add(self, run_id: int, embedding: np.ndarray) -> None:
        vec = embedding.astype("float32").reshape(1, -1)
        self.index.add(vec)
        self.run_ids.append(run_id)

    def search(self, embedding: np.ndarray, k: int = 3) -> list[tuple[int, float]]:
        if self.index.ntotal == 0:
            return []
        k = min(k, self.index.ntotal)
        vec = embedding.astype("float32").reshape(1, -1)
        distances, indices = self.index.search(vec, k)
        return [
            (self.run_ids[idx], float(dist))
            for idx, dist in zip(indices[0], distances[0])
            if idx != -1
        ]

def _squash(x: float, scale: float = 1.0) -> float:
    """Bound an arbitrary real number to (-1, 1) via tanh, so no single
    feature can dominate the vector's magnitude just because its raw scale
    (e.g. row count) is bigger than another feature's (e.g. a 0-1 ratio)."""
    return float(np.tanh(x / scale))


def _hash_index(key: str, dim: int) -> int:
    """Deterministic feature-name -> vector-slot mapping (the hashing trick),
    so each named feature always lands in the same slot across calls/runs."""
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % dim


def _hash_sign(key: str) -> float:
    """Deterministic +1/-1 per feature name, so two features that happen to
    hash into the same slot partially cancel instead of always reinforcing."""
    return 1.0 if int(hashlib.md5((key + "|sign").encode()).hexdigest(), 16) % 2 == 0 else -1.0


def _add_feature(vec: np.ndarray, dim: int, name: str, value: float) -> None:
    idx = _hash_index(name, dim)
    vec[idx] += _hash_sign(name) * value

def embed_dataset_summary(summary: dict[str, Any], dim: int) -> np.ndarray:
    """Build a fixed-length numeric embedding from a dataset summary using
    the hashing trick: each named feature (numeric or categorical) is scaled
    to a bounded range, then added into a deterministic hashed slot of a
    `dim`-length vector. Unlike the previous tile+phase approach, every slot
    receives a genuinely different combination of features instead of a
    repeated pattern of the same 7 numbers.
    """
    vec = np.zeros(dim, dtype="float32")

    _add_feature(vec, dim, "n_rows", _squash(np.log1p(summary.get("n_rows", 0)), scale=5.0))
    _add_feature(vec, dim, "n_cols", _squash(np.log1p(summary.get("n_cols", 0)), scale=3.0))
    _add_feature(vec, dim, "pct_missing", _squash(summary.get("pct_missing", 0.0), scale=0.5))
    _add_feature(vec, dim, "imbalance_ratio", _squash(np.log1p(summary.get("imbalance_ratio", 1.0)), scale=2.0))

    n_cols = max(summary.get("n_cols", 1), 1)
    _add_feature(vec, dim, "categorical_ratio", summary.get("n_categorical", 0) / n_cols)
    _add_feature(vec, dim, "numerical_ratio", summary.get("n_numerical", 0) / n_cols)
    _add_feature(vec, dim, "avg_cardinality", _squash(np.log1p(summary.get("avg_cardinality", 0.0)), scale=3.0))

    # Previously discarded fields — now actually used:
    _add_feature(vec, dim, f"problem_type={summary.get('problem_type', 'unknown')}", 1.0)

    class_counts = summary.get("class_counts") or {}
    if class_counts:
        _add_feature(vec, dim, "n_classes", _squash(len(class_counts), scale=5.0))

    high_card_cols = summary.get("high_cardinality_columns") or []
    _add_feature(vec, dim, "n_high_cardinality_cols", _squash(len(high_card_cols), scale=3.0))

    corr_info = summary.get("correlation_characteristics") or {}
    top_pairs = corr_info.get("top_pairs") or []
    if top_pairs:
        strongest_r = max(abs(p.get("r", 0.0)) for p in top_pairs)
        _add_feature(vec, dim, "strongest_correlation", strongest_r)
        _add_feature(vec, dim, "n_correlated_pairs", _squash(len(top_pairs), scale=3.0))

    outlier_summary = summary.get("outlier_summary") or {}
    if outlier_summary:
        n_rows = max(summary.get("n_rows", 1), 1)
        total_outliers = sum(outlier_summary.values())
        _add_feature(vec, dim, "outlier_fraction", _squash(total_outliers / n_rows, scale=0.2))
        _add_feature(vec, dim, "n_outlier_columns",
                     _squash(sum(1 for v in outlier_summary.values() if v > 0), scale=3.0))

    norm = np.linalg.norm(vec)
    return (vec / norm).astype("float32") if norm > 0 else vec