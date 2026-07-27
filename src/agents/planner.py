"""Planner Agent.

Extracted verbatim from Notebook Cell 24 ("CELL 6i — PLANNER AGENT").

``planner_node`` takes ``llm_client``, ``structured``, and ``semantic`` as
explicit parameters (Stage E5 DI decisions):
  - ``llm_client``: mock calls preserved exactly; Stage F will inject the
    real ``LLMClient`` here.
  - ``structured``/``semantic``: only used on the fallback path (when
    ``state["planner_context"]`` is absent — i.e. a standalone run without
    Memory Retrieval), matching the same DI treatment already approved for
    ``retrieve_similar_runs`` (Stage E4, Issue A).

NOTE: this module imports ``CLASSIFICATION_MODELS``/``REGRESSION_MODELS``
from ``src.agents.model_recommendation`` (Stage E5.6) — an existing cross-cell
dependency in the notebook itself (Cell 24 already used these same
constants, defined in Cell 8), not something introduced by extraction.
"""
from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path 

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.agents.model_recommendation import CLASSIFICATION_MODELS, REGRESSION_MODELS
from src.config.settings import Config
from src.core.metadata import _infer_target_column
from src.memory.dataset_summary import retrieve_similar_runs
from src.memory.repository import MemoryRepository
from src.memory.semantic_store import SemanticMemory, embed_dataset_summary
from src.memory.structured_store import StructuredMemory
from src.models.agent_io import PlannerDecision
from src.models.state import PipelineState


def _quick_raw_dataframe_stats(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    feature_df = df.drop(columns=[target_column])
    numerical_columns = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [c for c in feature_df.columns if c not in numerical_columns]
    n_rows, n_cols = df.shape[0], feature_df.shape[1]
    pct_missing = float(df.isna().mean().mean())
    cardinalities = [int(feature_df[c].nunique(dropna=True)) for c in categorical_columns]
    avg_cardinality = float(np.mean(cardinalities)) if cardinalities else 0.0
    max_cardinality = int(max(cardinalities)) if cardinalities else 0

    target = df[target_column]
    is_numeric_target = pd.api.types.is_numeric_dtype(target)
    n_unique_target = int(target.nunique(dropna=True))
    looks_like_classification = (not is_numeric_target) or (n_unique_target <= max(20, int(0.05 * n_rows)))

    imbalance_ratio = 1.0
    if looks_like_classification:
        counts = target.value_counts(dropna=True)
        if len(counts) >= 2:
            imbalance_ratio = float(counts.iloc[0] / max(counts.iloc[-1], 1))

    return {
        "n_rows": n_rows, "n_cols": n_cols, "pct_missing": pct_missing,
        "n_numerical": len(numerical_columns), "n_categorical": len(categorical_columns),
        "avg_cardinality": avg_cardinality, "max_cardinality": max_cardinality,
        "looks_like_classification": looks_like_classification, "imbalance_ratio": imbalance_ratio,
    }


def _mock_planner_decision(
    fingerprint: dict[str, Any],
    similar_runs: list[dict[str, Any]],
    cfg: Config,
    critic: Optional[dict[str, Any]] = None,
    prior_best_model: Optional[str] = None,
) -> PlannerDecision:
    problem_type = "classification" if fingerprint["looks_like_classification"] else "regression"
    imputation_strategy = "median" if fingerprint["n_numerical"] > 0 else (
        "most_frequent" if fingerprint["n_categorical"] > 0 else "none"
    )
    if fingerprint["n_categorical"] == 0:
        categorical_encoding = "onehot"
    elif fingerprint["max_cardinality"] > cfg.high_cardinality_threshold:
        categorical_encoding = "mixed" if fingerprint["n_categorical"] > 1 else "ordinal"
    else:
        categorical_encoding = "onehot"
    scaling = "standard" if fingerprint["n_numerical"] > 0 else "none"
    use_pca = fingerprint["n_numerical"] > 15
    handle_imbalance = (
        problem_type == "classification" and fingerprint["imbalance_ratio"] > cfg.imbalance_ratio_threshold
    )

    per_categorical_width = 3 if fingerprint["max_cardinality"] <= cfg.high_cardinality_threshold else 1
    estimated_width = fingerprint["n_numerical"] + fingerprint["n_categorical"] * per_categorical_width
    feature_selection_k = min(20, estimated_width) if estimated_width > 25 else None

    pool = CLASSIFICATION_MODELS if problem_type == "classification" else REGRESSION_MODELS
    candidate_models = list(pool)
    if similar_runs:
        best_similar = similar_runs[0].get("chosen_model")
        if best_similar in candidate_models:
            candidate_models.remove(best_similar)
            candidate_models.insert(0, best_similar)

    revision_note = ""
    if critic is not None:
        flagged = (
            not critic.get("metrics_acceptable", True)
            or critic.get("overfitting_detected")
            or critic.get("leakage_suspected")
        )
        if prior_best_model and flagged and prior_best_model in candidate_models:
            candidate_models.remove(prior_best_model)
            revision_note = (
                f" Revision: excluding '{prior_best_model}' after Critic flagged it "
                f"({', '.join(critic.get('issues', [])[:2]) or critic.get('comments', '')})."
            )
        if critic.get("overfitting_detected"):
            use_pca = True

    reasoning = (
        f"Cheap fingerprint: {fingerprint['n_rows']} rows, {fingerprint['n_numerical']} numerical / "
        f"{fingerprint['n_categorical']} categorical cols, {fingerprint['pct_missing']:.1%} missing, "
        f"imbalance_ratio={fingerprint['imbalance_ratio']:.2f}. "
        + (f"Retrieved {len(similar_runs)} similar past run(s) from memory." if similar_runs
           else "No similar past runs found in memory.")
        + revision_note
    )

    return PlannerDecision(
        problem_type=problem_type, imputation_strategy=imputation_strategy,
        categorical_encoding=categorical_encoding, scaling=scaling, use_pca=use_pca,
        handle_imbalance=handle_imbalance, feature_selection_k=feature_selection_k,
        candidate_models=candidate_models, reasoning=reasoning,
    )


def _fingerprint_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    cardinalities = list(metadata.get("cardinality", {}).values())
    return {
        "n_rows": metadata["n_rows"], "n_cols": metadata["n_cols"],
        "pct_missing": metadata["pct_missing"],
        "n_numerical": metadata["n_numerical"], "n_categorical": metadata["n_categorical"],
        "avg_cardinality": metadata["avg_cardinality"],
        "max_cardinality": max(cardinalities) if cardinalities else 0,
        "looks_like_classification": metadata["problem_type"] == "classification",
        "imbalance_ratio": metadata["imbalance_ratio"],
    }


def planner_node(
    state: PipelineState,
    llm_client: Any,
    cfg: Config,
    structured: StructuredMemory,
    semantic: SemanticMemory,
) -> PipelineState:
    """LangGraph node. Prefers the upstream Memory Retrieval Agent's
    planner_context (new); falls back to the original inline
    fingerprint+retrieval if the graph didn't run Memory Retrieval first
    (e.g. a standalone smoke test).
    """
    df = state["dataset"]
    target_column = state.get("target_column") or _infer_target_column(df)
    state["target_column"] = target_column

    planner_context = state.get("planner_context")
    if planner_context is not None:
        fingerprint = _fingerprint_from_metadata(state["metadata"])
        similar_runs = planner_context["top_similar_datasets"]
    else:  # unchanged fallback path — identical to the original cell
        fingerprint = _quick_raw_dataframe_stats(df, target_column)
        embedding = embed_dataset_summary(fingerprint, cfg.faiss_dim)
        similar_runs = retrieve_similar_runs(structured, semantic, embedding, k=3)

    prior_critic = state.get("critic")
    prior_best_model = state.get("metrics", {}).get("best_model_name") if prior_critic else None

    system_prompt = (
        "You are the Planner Agent in a multi-agent ML pipeline. Given dataset "
        "statistics and any similar past runs retrieved from memory, decide a "
        "first-pass preprocessing and candidate-model strategy. Return structured JSON."
    )
    user_prompt = json.dumps(
        {"fingerprint": fingerprint, "similar_runs": similar_runs, "prior_critic": prior_critic}, default=str
    )
    decision = llm_client.structured_call(
        schema=PlannerDecision, system_prompt=system_prompt, user_prompt=user_prompt,
        mock_fn=lambda: _mock_planner_decision(
            fingerprint, similar_runs, cfg, critic=prior_critic, prior_best_model=prior_best_model
        ),
    )

    state["planner_decision"] = decision.model_dump()
    state.setdefault("history", []).append({
        "node": "planner", "status": "ok",
        "iteration": state.get("iteration", 0), "problem_type": decision.problem_type,
    })
    return state