"""Memory Retrieval Agent — LangGraph node.

Extracted verbatim from Notebook Cell 23
("NEW CELL — MEMORY RETRIEVAL AGENT (LangGraph node)").

``memory_repository`` is an explicit parameter here, not a module global —
same DI treatment already approved for ``retrieve_similar_runs`` (Stage E4,
Issue A). No retrieval logic changed.
"""
from __future__ import annotations
import sys
import os
from pathlib import Path 

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.memory.dataset_summary import build_semantic_dataset_summary
from src.memory.planner_context import build_planner_context
from src.memory.ranking import rank_memories
from src.memory.repository import MemoryRepository
from src.models.state import PipelineState


def memory_retrieval_node(state: PipelineState, memory_repository: MemoryRepository, cfg) -> PipelineState:
    """LangGraph node: summarize -> embed -> retrieve -> rank -> build Planner
    context. Runs after Dataset Analyzer, before Planner. Does not call or
    modify the Planner in any way.
    """
    metadata = state["metadata"]
    df = state["dataset"]

    dataset_summary = build_semantic_dataset_summary(metadata, df)
    candidates = memory_repository.retrieve_memories(
        dataset_summary, k=cfg.memory_retrieval_top_k, min_similarity=cfg.memory_min_similarity,
    )
    ranked = rank_memories(candidates)
    planner_context = build_planner_context(dataset_summary, ranked)

    state["dataset_summary"] = dataset_summary
    state["retrieved_memories"] = ranked
    state["retrieval_scores"] = {m["run_id"]: m["usefulness"] for m in ranked}
    state["planner_context"] = planner_context
    state.setdefault("history", []).append({
        "node": "memory_retrieval", "status": "ok", "n_retrieved": len(ranked),
    })
    return state