"""LangGraph pipeline state schema.

Extracted verbatim from Notebook Cell 5 ("CELL 5 — LANGGRAPH STATE") — the
``PipelineState`` TypedDict only. The five pydantic agent-output schemas
that were defined in the same notebook cell are extracted separately into
``models/agent_io.py``: they have a different consumer pattern (one schema
per agent, validated per-call) than ``PipelineState`` (one shared dict
threaded through the entire graph), so they're kept in separate files per
the Cell-to-Module Mapping agreed during the modularization analysis.
"""
from __future__ import annotations

from typing import Any, TypedDict

import pandas as pd


class PipelineState(TypedDict, total=False):
    dataset: pd.DataFrame
    dataset_source: str
    target_column: str
    metadata: dict[str, Any]
    dataset_summary: dict[str, Any]          # NEW
    retrieved_memories: list[dict[str, Any]] # NEW
    retrieval_scores: dict[int, float]       # NEW
    planner_context: dict[str, Any]          # NEW
    planner_decision: dict[str, Any]
    eda: dict[str, Any]
    features: dict[str, Any]
    model_recommendation: dict[str, Any]
    models: dict[str, Any]
    metrics: dict[str, Any]
    critic: dict[str, Any]
    report: str
    history: list[dict[str, Any]]
    iteration: int
    needs_revision: bool
    current_experience: dict[str, Any]        # NEW
    experience_score: dict[str, Any]          # NEW
    memory_update_decision: dict[str, Any]    # NEW
    # memory_quality removed — Implementation Task 5, persistence audit: this
    # PipelineState key was write-only (set by memory_update_policy_node, never
    # read by any node or by report generation). The *persisted* memory_quality
    # column on the `runs` table (a different thing — the curated-memory quality
    # score written via MemoryRepository.set_quality and read back via
    # best_match.get("memory_quality") when deciding replace/merge/ignore) is
    # unaffected and still fully used; only the redundant in-run state copy is gone.
    similar_memories: list[dict[str, Any]]    # NEW
    update_reason: str                        # NEW
    explainability: dict[str, Any]            # NEW — producer: shap_explainability_node (Cell 10);
                                               # consumers: critic_agent_node (Cell 6g), build_experience
                                               # (Task 6), generate_markdown_report (Cell 11). Holds only
                                               # the compact ExplainabilitySummary (Task 5) — never raw
                                               # SHAP tensors.