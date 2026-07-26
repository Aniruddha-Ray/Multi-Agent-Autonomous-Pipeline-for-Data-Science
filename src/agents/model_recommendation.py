"""Model Recommendation Agent.

Extracted verbatim from Notebook Cell 8 ("CELL 6d — MODEL RECOMMENDATION AGENT").
``model_recommendation_node`` takes ``llm_client`` as an explicit parameter
(Stage E5 DI decision).
"""
from __future__ import annotations

import json
from typing import Any
import sys
import os
from pathlib import Path 

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.models.agent_io import ModelRecommendation
from src.models.state import PipelineState

CLASSIFICATION_MODELS: list[str] = [
    "Random Forest", "XGBoost", "LightGBM", "CatBoost", "Logistic Regression", "SVM",
]
REGRESSION_MODELS: list[str] = [
    "Random Forest", "XGBoost", "LightGBM", "CatBoost", "Linear Regression", "ElasticNet", "SVM",
]

MODEL_TRAITS: dict[str, dict[str, bool]] = {
    "Random Forest":       {"tree_based": True,  "native_categorical": False, "needs_scaling": False},
    "XGBoost":              {"tree_based": True,  "native_categorical": False, "needs_scaling": False},
    "LightGBM":              {"tree_based": True,  "native_categorical": False, "needs_scaling": False},
    "CatBoost":              {"tree_based": True,  "native_categorical": True,  "needs_scaling": False},
    "Logistic Regression": {"tree_based": False, "native_categorical": False, "needs_scaling": True},
    "Linear Regression":   {"tree_based": False, "native_categorical": False, "needs_scaling": True},
    "ElasticNet":            {"tree_based": False, "native_categorical": False, "needs_scaling": True},
    "SVM":                    {"tree_based": False, "native_categorical": False, "needs_scaling": True},
}


def _score_model_candidates(metadata: dict[str, Any]) -> dict[str, float]:
    problem_type = metadata["problem_type"]
    n_rows = metadata["n_rows"]
    n_categorical = metadata["n_categorical"]
    high_card = bool(metadata["high_cardinality_columns"])
    imbalanced = metadata["is_imbalanced"]
    n_cols = metadata["n_numerical"] + metadata["n_categorical"]

    candidates = CLASSIFICATION_MODELS if problem_type == "classification" else REGRESSION_MODELS
    scores: dict[str, float] = {}
    for model_name in candidates:
        traits = MODEL_TRAITS[model_name]
        score = 1.0
        if traits["tree_based"]:
            score += 1.5
            if high_card or n_categorical > 0:
                score += 1.0
            if imbalanced and model_name in ("XGBoost", "LightGBM", "CatBoost"):
                score += 0.5
            if n_rows > 10_000 and model_name == "LightGBM":
                score += 0.5
            if n_categorical >= 3 and model_name == "CatBoost":
                score += 0.5
        else:
            if model_name in ("Logistic Regression", "Linear Regression", "ElasticNet"):
                score += 0.5
                if n_cols <= 15:
                    score += 0.5
                if n_rows < 500:
                    score += 0.3
            if model_name == "SVM":
                score += 0.7 if n_rows <= 3000 else -1.5
        scores[model_name] = round(score, 2)
    return scores


def _reason_for_model(model_name: str, metadata: dict[str, Any]) -> str:
    traits = MODEL_TRAITS[model_name]
    n_rows = metadata["n_rows"]
    high_card = bool(metadata["high_cardinality_columns"])
    imbalanced = metadata["is_imbalanced"]
    n_cols = metadata["n_numerical"] + metadata["n_categorical"]

    if traits["tree_based"]:
        parts = ["Gradient/ensemble tree model — robust to feature scale and nonlinear interactions."]
        if high_card:
            parts.append("Tolerates the high-cardinality categorical column(s) better than one-hot-heavy linear models.")
        if imbalanced and model_name in ("XGBoost", "LightGBM", "CatBoost"):
            parts.append("Supports native class-weighting to counter the observed target imbalance.")
        if model_name == "CatBoost":
            parts.append("Best native categorical-feature handling of the ensemble family.")
        if model_name == "LightGBM" and n_rows > 10_000:
            parts.append(f"Histogram-based training scales efficiently to {n_rows} rows.")
        return " ".join(parts)

    if model_name == "SVM":
        return (
            f"Kernel methods can capture nonlinear boundaries on small-to-medium data "
            f"({n_rows} rows), but training cost grows poorly beyond a few thousand rows."
        )

    return (
        f"Simple, interpretable, fast-to-train baseline; well suited to a "
        f"{'low' if n_cols <= 15 else 'moderate'}-dimensional ({n_cols} feature) "
        f"{'and small-sample ' if n_rows < 500 else ''}setting where linear decision boundaries are a reasonable prior."
    )


def _mock_model_recommendation(metadata: dict[str, Any]) -> ModelRecommendation:
    scores = _score_model_candidates(metadata)
    ranked_models = sorted(scores, key=lambda m: scores[m], reverse=True)
    reasoning = {
        model_name: f"(score={scores[model_name]}) {_reason_for_model(model_name, metadata)}"
        for model_name in ranked_models
    }
    return ModelRecommendation(ranked_models=ranked_models, reasoning=reasoning)


def model_recommendation_node(state: PipelineState, llm_client: Any) -> PipelineState:
    metadata = state["metadata"]
    fe_decision = state["features"]["decision"]

    system_prompt = (
        "You are the Model Recommendation Agent in a multi-agent ML pipeline. "
        "Choose among Random Forest, XGBoost, LightGBM, CatBoost, Logistic Regression, "
        "Linear Regression, ElasticNet, SVM (restricted to those valid for the problem "
        "type). Rank them and explain WHY each is/isn't suitable. Return structured JSON."
    )
    user_prompt = json.dumps(
        {
            "metadata": {k: v for k, v in metadata.items() if k != "summary_statistics"},
            "feature_engineering_decision": fe_decision,
        },
        default=str,
    )
    recommendation = llm_client.structured_call(
        schema=ModelRecommendation,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        mock_fn=lambda: _mock_model_recommendation(metadata),
    )

    planner_candidates = state.get("planner_decision", {}).get("candidate_models")
    if planner_candidates:
        filtered = [m for m in recommendation.ranked_models if m in planner_candidates]
        if filtered:
            recommendation = ModelRecommendation(ranked_models=filtered, reasoning=recommendation.reasoning)

    state["model_recommendation"] = recommendation.model_dump()
    state.setdefault("history", []).append({"node": "model_recommendation_agent", "status": "ok"})
    return state