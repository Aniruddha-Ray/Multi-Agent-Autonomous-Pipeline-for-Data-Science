"""Memory Update Policy.

Extracted verbatim from Notebook Cell 16 ("NEW CELL — MEMORY UPDATE POLICY").
``memory_repository`` is an explicit parameter (Stage E5 DI decision, same
treatment as ``memory_retrieval_node`` and ``retrieve_similar_runs``).
"""
from __future__ import annotations

from typing import Any
import sys
import os
from pathlib import Path 

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.memory.repository import MemoryRepository
from src.models.experience import build_experience_payload
from src.models.state import PipelineState

REPLACE_SIMILARITY_THRESHOLD = 0.90
MERGE_QUALITY_TOLERANCE = 0.05
MIN_EXPERIENCE_SCORE_TO_STORE = 0.40


def _memory_quality(scored: dict[str, Any]) -> float:
    return round(0.6 * scored["experience_score"] + 0.4 * scored["generalization_score"], 4)


def memory_update_policy_node(state: PipelineState, memory_repository: MemoryRepository) -> PipelineState:
    experience = state["current_experience"]
    scored = state["experience_score"]
    dataset_summary = experience["dataset_summary"]
    quality = _memory_quality(scored)

    if scored["recommendation"] == "discard" or scored["experience_score"] < MIN_EXPERIENCE_SCORE_TO_STORE:
        state["memory_update_decision"] = {
            "action": "ignore", "reason": scored["reasoning"],
            "total_runs_stored": memory_repository.structured.count(),
        }
        state["similar_memories"] = []
        state["update_reason"] = f"Discarded: {scored['reasoning']}"
        state.setdefault("history", []).append({"node": "memory_update_policy", "status": "ok", "action": "ignore"})
        return state

    similar = memory_repository.retrieve_memories(dataset_summary, k=1, min_similarity=REPLACE_SIMILARITY_THRESHOLD)
    state["similar_memories"] = similar

    critic_notes = experience["critic_feedback"]
    chosen_model = experience["pipeline_configuration"].get("model")

    experience_payload = build_experience_payload(experience, experience_score=scored["experience_score"])

    run_id_to_report = None

    if not similar:
        run_id = memory_repository.save_memory(
            dataset_summary, planner_reasoning=experience["planner_reasoning"],
            chosen_model=chosen_model, metrics=experience["evaluation_metrics"], critic_notes=critic_notes,
            experience_payload=experience_payload,
        )
        memory_repository.set_quality(run_id, memory_quality=quality, experience_score=scored["experience_score"],
                                       confidence=scored["confidence"], success_rate=1.0 if scored["recommendation"] == "retain" else 0.0)
        decision = {"action": "store_new", "run_id": run_id, "reason": "No sufficiently similar memory found."}
        run_id_to_report = run_id
    else:
        best_match = similar[0]
        existing_quality = best_match.get("memory_quality") or 0.0
        if quality > existing_quality + MERGE_QUALITY_TOLERANCE:
            memory_repository.delete_memory(best_match["run_id"])
            run_id = memory_repository.save_memory(
                dataset_summary, planner_reasoning=experience["planner_reasoning"],
                chosen_model=chosen_model, metrics=experience["evaluation_metrics"], critic_notes=critic_notes,
                experience_payload=experience_payload,
            )
            memory_repository.set_quality(run_id, memory_quality=quality, experience_score=scored["experience_score"],
                                           confidence=scored["confidence"], success_rate=1.0)
            decision = {"action": "replace", "replaced_run_id": best_match["run_id"], "new_run_id": run_id,
                        "reason": f"New quality {quality:.3f} > existing {existing_quality:.3f}"}
            run_id_to_report = run_id
        elif abs(quality - existing_quality) <= MERGE_QUALITY_TOLERANCE:
            memory_repository.update_memory(best_match["run_id"], chosen_model=chosen_model,
                                             metrics_json=experience["evaluation_metrics"], critic_notes_json=critic_notes,
                                             experience_payload_json=experience_payload)
            memory_repository.set_quality(best_match["run_id"], memory_quality=max(quality, existing_quality),
                                           experience_score=scored["experience_score"], confidence=scored["confidence"])
            decision = {"action": "merge", "run_id": best_match["run_id"],
                        "reason": f"Comparable quality ({quality:.3f} vs {existing_quality:.3f}) — merged in place."}
            run_id_to_report = best_match["run_id"]
        else:
            decision = {"action": "ignore", "reason": f"Existing memory (quality {existing_quality:.3f}) already stronger than new ({quality:.3f})."}

    decision["total_runs_stored"] = memory_repository.structured.count()
    if run_id_to_report:
        decision["reported_run_id"] = run_id_to_report

    state["memory_update_decision"] = decision
    state["update_reason"] = decision["reason"]
    state.setdefault("history", []).append({"node": "memory_update_policy", "status": "ok", **decision})
    return state