"""Embedding provider abstraction for memory retrieval.

Extracted verbatim from Notebook Cell 19
("NEW CELL — MEMORY RETRIEVAL: EMBEDDING PROVIDER ABSTRACTION"). The
Planner/Memory Retrieval layer only ever talks to the ``EmbeddingProvider``
interface — it never knows which concrete embedding backend is active.

NOT included in this module: ``embedding_provider = get_embedding_provider(CFG)``
— that instantiation is deferred to the Stage E8 composition root.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import sys
import os
from pathlib import Path 

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.settings import Config
from src.memory.semantic_store import embed_dataset_summary


class EmbeddingProvider(ABC):
    """The Planner/Memory Retrieval layer only ever talks to this interface —
    it never knows which concrete embedding backend is active."""

    @abstractmethod
    def embed(self, summary: dict[str, Any]) -> np.ndarray: ...

    @property
    @abstractmethod
    def dim(self) -> int: ...


class LocalHeuristicEmbeddingProvider(EmbeddingProvider):
    """Wraps ``embed_dataset_summary`` unchanged — today's default."""

    def __init__(self, dim: int) -> None:
        self._dim = dim

    def embed(self, summary: dict[str, Any]) -> np.ndarray:
        return embed_dataset_summary(summary, self._dim)

    @property
    def dim(self) -> int:
        return self._dim


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Not wired up in this stage — placeholder for the future migration."""

    def __init__(self, dim: int = 1536) -> None:
        self._dim = dim

    def embed(self, summary: dict[str, Any]) -> np.ndarray:
        raise NotImplementedError("OpenAI embeddings are not enabled in Stage B.")

    @property
    def dim(self) -> int:
        return self._dim


class VoyageAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, dim: int = 1024) -> None:
        self._dim = dim

    def embed(self, summary: dict[str, Any]) -> np.ndarray:
        raise NotImplementedError("VoyageAI embeddings are not enabled in Stage B.")

    @property
    def dim(self) -> int:
        return self._dim


class GeminiEmbeddingProvider(EmbeddingProvider):
    def __init__(self, dim: int = 768) -> None:
        self._dim = dim

    def embed(self, summary: dict[str, Any]) -> np.ndarray:
        raise NotImplementedError("Gemini embeddings are not enabled in Stage B.")

    @property
    def dim(self) -> int:
        return self._dim


_EMBEDDING_REGISTRY: dict[str, type[EmbeddingProvider]] = {
    "local": LocalHeuristicEmbeddingProvider,
    "openai": OpenAIEmbeddingProvider,
    "voyageai": VoyageAIEmbeddingProvider,
    "gemini": GeminiEmbeddingProvider,
}


def get_embedding_provider(cfg: Config) -> EmbeddingProvider:
    """Factory the Memory Retrieval Agent calls — swap providers via config only."""
    provider_cls = _EMBEDDING_REGISTRY.get(cfg.embedding_provider, LocalHeuristicEmbeddingProvider)
    return provider_cls(cfg.faiss_dim)