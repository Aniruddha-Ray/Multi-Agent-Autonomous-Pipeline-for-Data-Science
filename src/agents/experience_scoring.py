"""Experience Scoring.

Extracted verbatim from Notebook Cell 14 ("NEW CELL — EXPERIENCE SCORING: SCORER").
"""
from __future__ import annotations

from typing import Any

import numpy as np
import sys
import os
from pathlib import Path 

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.memory.ranking import _performance_score
from src.models.experience import Experience, build_experience
from src.models.state import PipelineState


def _robustness_score(cv_metrics: dict[str, Any]) -> float:
    scores = cv_metrics.get("fold_scores") if cv_metrics else None
    if not scores or len(scores) < 2:
        return 0.5
    mean, std = float(np.mean(scores)), float(np.std(scores))
    if mean == 0:
        return 0.5
    return max(0.0, min(1.0, 1.0 - (std / abs(mean))))


def _overfitting_penalty(critic_feedback: dict[str, Any]) -> float:
    if "overfitting_detected" in critic_feedback:
        return 0.2 if critic_feedback["overfitting_detected"] else 1.0
    return 0.5


def _preprocessing_quality(critic_feedback: dict[str, Any]) -> float:
    if "feature_engineering_ok" in critic_feedback:
        return 1.0 if critic_feedback["feature_engineering_ok"] else 0.3
    return 0.7


WEIGHTS = {
    "performance": 0.35, "robustness": 0.20, "critic_confidence": 0.15,
    "preprocessing_quality": 0.10, "planner_confidence": 0.10, "overfitting": 0.10,
}


def score_experience(experience: Experience) -> dict[str, Any]:
    performance = _performance_score(experience.evaluation_metrics)
    robustness = _robustness_score(experience.cv_metrics)
    overfitting = _overfitting_penalty(experience.critic_feedback)
    preprocessing_quality = _preprocessing_quality(experience.critic_feedback)
    critic = experience.critic_feedback
    critic_confidence = (
        0.9 if critic.get("recommendation") == "approve" and critic.get("metrics_acceptable", True)
        else 0.3 if critic.get("recommendation") == "revise"
        else 0.5
    )
    planner_confidence = 0.5

    experience_score = (
        WEIGHTS["performance"] * performance
        + WEIGHTS["robustness"] * robustness
        + WEIGHTS["critic_confidence"] * critic_confidence
        + WEIGHTS["preprocessing_quality"] * preprocessing_quality
        + WEIGHTS["planner_confidence"] * planner_confidence
        + WEIGHTS["overfitting"] * overfitting
    )
    generalization_score = round((robustness + overfitting) / 2.0, 4)
    confidence = round((critic_confidence + planner_confidence) / 2.0, 4)

    if experience_score >= 0.7 and generalization_score >= 0.6:
        recommendation = "retain"
    elif experience_score < 0.4 or generalization_score < 0.3:
        recommendation = "discard"
    else:
        recommendation = "borderline"

    reasoning = (
        f"performance={performance:.2f}, robustness={robustness:.2f}, "
        f"overfitting_penalty={overfitting:.2f}, critic_confidence={critic_confidence:.2f}, "
        f"planner_confidence={planner_confidence:.2f}, preprocessing_quality={preprocessing_quality:.2f} "
        f"=> experience_score={experience_score:.3f} ({recommendation})"
    )
    return {
        "experience_score": round(experience_score, 4),
        "confidence": confidence,
        "generalization_score": generalization_score,
        "recommendation": recommendation,
        "reasoning": reasoning,
    }


def experience_scorer_node(state: PipelineState) -> PipelineState:
    execution_time = state.get("history", [{}])[-1].get("elapsed_seconds", 0.0) if state.get("history") else 0.0
    experience = build_experience(state, execution_time)
    scored = score_experience(experience)

    state["current_experience"] = experience.to_dict()
    state["experience_score"] = scored
    state.setdefault("history", []).append({"node": "experience_scorer", "status": "ok", **scored})
    return state