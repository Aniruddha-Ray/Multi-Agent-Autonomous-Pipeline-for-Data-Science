"""FAISS-backed semantic index over dataset-summary embeddings.

Extracted verbatim from Notebook Cell 3 ("CELL 4 — MEMORY INITIALIZATION")
— the ``SemanticMemory`` class and ``embed_dataset_summary`` function only.
"""
from __future__ import annotations

from typing import Any

import faiss
import numpy as np


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


def embed_dataset_summary(summary: dict[str, Any], dim: int) -> np.ndarray:
    """Build a fixed-length numeric embedding from a dataset summary.

    L2-normalized (Task 4, architecture audit) so every embedding — whether
    produced here for storage (``MemoryRepository.save_memory``) or for a
    query (``MemoryRepository.retrieve_memories``, the Planner's fallback
    path) — is a unit vector. FAISS's ``IndexFlatL2`` (unchanged) then gives
    a distance that is monotonically related to cosine similarity for every
    vector in the index, not just at query time: ``cos = 1 - d^2/2``. Without
    this, only the query vector was ever normalized, so stored vectors and
    query vectors weren't on a consistent footing and the "cosine similarity"
    computed downstream was not actually correct.
    """
    base_features = np.array(
        [
            np.log1p(summary.get("n_rows", 0)),
            np.log1p(summary.get("n_cols", 0)),
            summary.get("pct_missing", 0.0),
            np.log1p(summary.get("imbalance_ratio", 1.0)),
            summary.get("n_categorical", 0) / max(summary.get("n_cols", 1), 1),
            summary.get("n_numerical", 0) / max(summary.get("n_cols", 1), 1),
            np.log1p(summary.get("avg_cardinality", 0.0)),
        ],
        dtype="float32",
    )
    reps = int(np.ceil(dim / len(base_features)))
    tiled = np.tile(base_features, reps)[:dim]
    phase = np.arange(dim, dtype="float32") * 0.01
    vec = (tiled + phase).astype("float32")

    norm = np.linalg.norm(vec)
    return (vec / norm).astype("float32") if norm > 0 else vec