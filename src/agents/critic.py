"""Critic Agent.

Extracted verbatim from Notebook Cell 11 ("CELL 6g — CRITIC AGENT").
``critic_agent_node`` takes ``llm_client`` as an explicit parameter
(Stage E5 DI decision).
"""
from __future__ import annotations

import json
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os
from pathlib import Path 

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.agents.training import compute_metrics
from src.config.settings import CFG
from src.models.agent_io import CriticReview
from src.models.state import PipelineState

MIN_ACCEPTABLE_F1: float = 0.55
MIN_ACCEPTABLE_R2: float = 0.3


def _max_feature_target_correlation(
    df: pd.DataFrame, target_column: str, numerical_columns: list[str], problem_type: str
) -> float:
    if not numerical_columns:
        return 0.0
    y = df[target_column]
    if problem_type == "classification":
        if y.nunique() != 2:
            return 0.0
        y_numeric = pd.factorize(y)[0].astype(float)
    else:
        y_numeric = y.astype(float).values

    correlations = []
    for col in numerical_columns:
        filled = df[col].fillna(df[col].mean())
        if filled.std() == 0:
            continue
        correlations.append(abs(np.corrcoef(filled, y_numeric)[0, 1]))
    return float(np.nanmax(correlations)) if correlations else 0.0


def _feature_engineering_review(decision: dict[str, Any], metadata: dict[str, Any]) -> tuple[bool, list[str]]:
    issues: list[str] = []
    if decision["encoding_strategy"] == "onehot" and metadata["high_cardinality_columns"]:
        issues.append(
            f"One-hot encoding chosen despite high-cardinality column(s) "
            f"{metadata['high_cardinality_columns']} — risk of dimensionality blow-up."
        )
    if decision["use_pca"] and metadata["n_numerical"] < 2:
        issues.append("PCA enabled with fewer than 2 numerical features — unlikely to be useful.")
    total_features = metadata["n_numerical"] + metadata["n_categorical"]
    if decision.get("feature_selection_k") and decision["feature_selection_k"] >= total_features:
        issues.append("SelectKBest k is >= total feature count — feature selection has no effect.")
    return (len(issues) == 0, issues)


def _mock_critic_review(state: PipelineState) -> CriticReview:
    metadata = state["metadata"]
    decision = state["features"]["decision"]
    problem_type = metadata["problem_type"]
    target_column = state["target_column"]
    df = state["dataset"]
    best_model_name = state["metrics"]["best_model_name"]
    best_pipeline = state["models"]["trained_pipelines"][best_model_name]

    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CFG.test_size, random_state=CFG.random_state,
        stratify=y if problem_type == "classification" else None,
    )

    y_train_pred = best_pipeline.predict(X_train)
    y_train_proba = best_pipeline.predict_proba(X_train) if hasattr(best_pipeline, "predict_proba") else None
    train_metrics = compute_metrics(y_train.values, y_train_pred, y_train_proba, problem_type)
    test_metrics = state["metrics"]["best_model_metrics"]

    metric_key = "f1" if problem_type == "classification" else "r2"
    gap = train_metrics[metric_key] - test_metrics[metric_key]
    overfitting_detected = gap > CFG.overfit_gap_threshold

    max_corr = _max_feature_target_correlation(df, target_column, metadata["numerical_columns"], problem_type)
    near_perfect = (
        (test_metrics.get("f1", 0.0) > 0.98 or test_metrics.get("roc_auc", 0.0) > 0.995)
        if problem_type == "classification"
        else test_metrics.get("r2", 0.0) > 0.995
    )
    leakage_suspected = bool(near_perfect and max_corr > 0.95)

    explainability = state.get("explainability", {})
    if explainability.get("suspected_target_leakage"):
        leakage_suspected = True

    feature_engineering_ok, fe_issues = _feature_engineering_review(decision, metadata)

    planner_decision_for_review = state.get("planner_decision", {}) or {}
    planner_problem_type = planner_decision_for_review.get("problem_type")
    planner_contract_violation = bool(planner_problem_type and planner_problem_type != problem_type)

    metrics_acceptable = (
        test_metrics.get("f1", 0.0) >= MIN_ACCEPTABLE_F1
        if problem_type == "classification"
        else test_metrics.get("r2", 0.0) >= MIN_ACCEPTABLE_R2
    )

    issues = list(fe_issues)
    if overfitting_detected:
        issues.append(
            f"Train/test gap of {gap:.3f} on '{metric_key}' exceeds the "
            f"{CFG.overfit_gap_threshold} threshold — possible overfitting."
        )
    if leakage_suspected:
        issues.append(
            f"Near-perfect held-out performance combined with a strong single-feature/target "
            f"correlation ({max_corr:.3f}) — possible data leakage."
        )
    if not metrics_acceptable:
        issues.append(f"Held-out '{metric_key}' does not clear the minimum acceptability bar.")

    explainability_concern = bool(
        explainability.get("dominant_feature_ratio", 0.0) > 0.6
        or explainability.get("feature_diversity_score", 1.0) < 0.3
    )
    for note in explainability.get("critic_explainability_notes", []):
        if "over-reliance" in note or "concentrated" in note or "leakage" in note:
            issues.append(note)

    if planner_contract_violation:
        issues.append(
            f"Planner recommended problem_type='{planner_problem_type}' but the Dataset "
            f"Analyzer determined problem_type='{problem_type}' from the target column — "
            "the Planner's preprocessing/model strategy may be based on a mismatched premise."
        )

    recommendation: Literal["approve", "revise"] = (
        "revise" if (
            overfitting_detected or leakage_suspected or not feature_engineering_ok
            or not metrics_acceptable or explainability_concern or planner_contract_violation
        ) else "approve"
    )

    comments = (
        f"Reviewed '{best_model_name}': train {metric_key}={train_metrics[metric_key]:.3f} vs "
        f"held-out {metric_key}={test_metrics[metric_key]:.3f} (gap={gap:.3f}). "
        + ("No material issues found; approving the pipeline." if recommendation == "approve"
           else "Issues found — recommending revision.")
    )

    return CriticReview(
        overfitting_detected=overfitting_detected,
        leakage_suspected=leakage_suspected,
        feature_engineering_ok=feature_engineering_ok,
        metrics_acceptable=metrics_acceptable,
        issues=issues,
        recommendation=recommendation,
        comments=comments,
    )


def critic_agent_node(state: PipelineState, llm_client: Any) -> PipelineState:
    system_prompt = (
        "You are the Critic Agent in a multi-agent ML pipeline. Review the "
        "preprocessing, chosen models, evaluation metrics, and explainability "
        "evidence. Detect overfitting, leakage, over-reliance on a single "
        "feature, ignored important features, suspicious importance "
        "distributions, or poor feature engineering. Return structured JSON "
        "with an 'approve' or 'revise' recommendation."
    )
    user_prompt = json.dumps(
        {
            "best_model": state["metrics"]["best_model_name"],
            "best_model_metrics": state["metrics"]["best_model_metrics"],
            "feature_engineering_decision": state["features"]["decision"],
            "explainability_summary": state.get("explainability", {}),
            "planner_decision": state.get("planner_decision", {}),
            "metadata_summary": {
                k: v for k, v in state["metadata"].items()
                if k not in ("summary_statistics", "correlation_matrix")
            },
        },
        default=str,
    )
    review = llm_client.structured_call(
        schema=CriticReview,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        mock_fn=lambda: _mock_critic_review(state),
    )

    state["critic"] = review.model_dump()
    state["needs_revision"] = review.recommendation == "revise"
    state["iteration"] = state.get("iteration", 0) + 1
    state.setdefault("history", []).append({
        "node": "critic_agent", "status": "ok", "recommendation": review.recommendation,
    })
    return state