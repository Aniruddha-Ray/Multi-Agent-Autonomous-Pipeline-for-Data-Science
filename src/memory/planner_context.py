"""Planner context builder — assembles retrieved memory into Planner input.

Extracted verbatim from Notebook Cell 22
("NEW CELL — MEMORY RETRIEVAL: PLANNER CONTEXT BUILDER").
"""
from __future__ import annotations

from typing import Any


def build_planner_context(
    dataset_summary: dict[str, Any],
    ranked_memories: list[dict[str, Any]],
    top_n: int = 3,
) -> dict[str, Any]:
    """Compact context handed to the Planner. The Planner's own reasoning
    is unchanged — only this input changes."""
    top = ranked_memories[:top_n]
    successful = [m for m in ranked_memories if m["quality_label"] == "success"][:top_n]
    failures = [m for m in ranked_memories if m["quality_label"] == "failure"][:2]

    model_votes: dict[str, float] = {}
    for m in successful:
        if m.get("chosen_model"):
            model_votes[m["chosen_model"]] = model_votes.get(m["chosen_model"], 0.0) + m["usefulness"]
    recommended_models = sorted(model_votes, key=lambda m: model_votes[m], reverse=True)

    return {
        "current_dataset_summary": dataset_summary,
        "top_similar_datasets": top,
        "successful_pipelines": successful,
        "failures_to_avoid": failures,
        "recommended_models": recommended_models,
    }