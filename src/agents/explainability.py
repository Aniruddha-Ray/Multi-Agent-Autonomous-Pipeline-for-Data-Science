"""SHAP Explainability Agent.

Extracted verbatim from Notebook Cell 25 ("CELL 10 — SHAP EXPLAINABILITY")
— everything except ``display_explainability_section``, which is a
presentation-layer function (IPython display + plot generation) and
belongs in ``reports/``, to be extracted in Stage E7.
"""
from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline
import sys
import os
from pathlib import Path 

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.config.settings import CFG
from src.models.state import PipelineState

SHAP_BACKGROUND_SIZE: int = 100
SHAP_SAMPLE_SIZE: int = 200
IGNORED_FEATURE_IMPORTANCE_RATIO: float = 0.01


def _get_post_preprocessing_feature_names(pipeline: Pipeline) -> list[str]:
    names: list[str] = list(pipeline.named_steps["preprocess"].get_feature_names_out())

    if "select" in pipeline.named_steps:
        mask = pipeline.named_steps["select"].get_support()
        names = [n for n, keep in zip(names, mask) if keep]

    if "pca" in pipeline.named_steps:
        n_components = pipeline.named_steps["pca"].n_components_
        names = [f"PC{i + 1}" for i in range(n_components)]

    return names


def _transform_up_to_model(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    transformed: Any = X
    for _, step in pipeline.steps[:-1]:
        transformed = step.transform(transformed)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    return np.asarray(transformed)


def _build_shap_explainer(model: Any, background: np.ndarray) -> Any:
    tree_based = model.__class__.__name__ in {
        "RandomForestClassifier", "RandomForestRegressor",
        "XGBClassifier", "XGBRegressor",
        "LGBMClassifier", "LGBMRegressor",
        "CatBoostClassifier", "CatBoostRegressor",
    }
    linear_based = model.__class__.__name__ in {
        "LogisticRegression", "LinearRegression", "ElasticNet",
    }

    if tree_based:
        return shap.TreeExplainer(model)
    if linear_based:
        return shap.LinearExplainer(model, background)
    predict_fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict
    return shap.KernelExplainer(predict_fn, background)


def _collapse_shap_values(shap_values: Any) -> np.ndarray:
    if isinstance(shap_values, list):
        return np.mean(np.abs(np.stack(shap_values, axis=0)), axis=0)
    values = np.asarray(shap_values)
    if values.ndim == 3:
        return np.mean(np.abs(values), axis=-1)
    return values


def run_shap_explainability(state: PipelineState, generate_plots: bool = True) -> dict[str, Any]:
    models = state["models"]
    decision = state["features"]["decision"]
    best_model_name = models["best_model_name"]
    pipeline = models["trained_pipelines"][best_model_name]
    X_test = models["X_test"]

    feature_names = _get_post_preprocessing_feature_names(pipeline)
    X_test_transformed = _transform_up_to_model(pipeline, X_test)

    rng = np.random.RandomState(CFG.random_state)
    n_background = min(SHAP_BACKGROUND_SIZE, X_test_transformed.shape[0])
    n_sample = min(SHAP_SAMPLE_SIZE, X_test_transformed.shape[0])
    background = X_test_transformed[rng.choice(X_test_transformed.shape[0], n_background, replace=False)]
    sample = X_test_transformed[rng.choice(X_test_transformed.shape[0], n_sample, replace=False)]

    model = pipeline.named_steps["model"]
    explainer = _build_shap_explainer(model, background)

    raw_shap_values = (
        explainer.shap_values(sample) if hasattr(explainer, "shap_values") else explainer(sample).values
    )
    shap_matrix = _collapse_shap_values(raw_shap_values)

    mean_abs_importance = np.abs(shap_matrix).mean(axis=0)
    order = np.argsort(mean_abs_importance)[::-1]
    ranked_features = [feature_names[i] for i in order]
    ranked_importance = [float(mean_abs_importance[i]) for i in order]

    artifact_paths: dict[str, str] = {}
    if generate_plots:
        import matplotlib.pyplot as plt

        summary_path = os.path.join(CFG.artifacts_dir, "shap_summary.png")
        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_matrix, sample, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(summary_path, dpi=110, bbox_inches="tight")
        plt.close()

        bar_path = os.path.join(CFG.artifacts_dir, "shap_feature_importance.png")
        top_k = min(15, len(ranked_features))
        fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * top_k)))
        ax.barh(ranked_features[:top_k][::-1], ranked_importance[:top_k][::-1], color="#C44E52")
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"Global Feature Importance — {best_model_name}")
        fig.tight_layout()
        fig.savefig(bar_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        artifact_paths = {"summary_plot": summary_path, "feature_importance_bar": bar_path}

    pca_used = bool(decision.get("use_pca"))
    observations = [
        f"Top driver of `{best_model_name}` predictions: **{ranked_features[0]}** "
        f"(mean |SHAP| = {ranked_importance[0]:.4f}).",
        f"Explained {n_sample} of {X_test_transformed.shape[0]} held-out rows using "
        f"{explainer.__class__.__name__} against a {n_background}-row background sample.",
    ]
    if pca_used:
        observations.append(
            "Feature engineering applied PCA, so SHAP attributions are over "
            "principal components (PC1, PC2, ...), not the original raw features."
        )

    return {
        "explainer_type": explainer.__class__.__name__,
        "best_model_name": best_model_name,
        "feature_names": feature_names,
        "ranked_features": ranked_features,
        "ranked_importance": ranked_importance,
        "n_background": n_background,
        "n_sample": n_sample,
        "artifact_paths": artifact_paths,
        "observations": observations,
        "pca_used": pca_used,
    }


def _dominant_feature_ratio(ranked_importance: list[float]) -> float:
    total = sum(ranked_importance)
    if total <= 0 or not ranked_importance:
        return 0.0
    return round(ranked_importance[0] / total, 4)


def _feature_diversity_score(ranked_importance: list[float]) -> float:
    total = sum(ranked_importance)
    if total <= 0 or len(ranked_importance) < 2:
        return 0.0
    probs = [v / total for v in ranked_importance if v > 0]
    entropy = -sum(p * math.log(p) for p in probs)
    max_entropy = math.log(len(ranked_importance))
    return round(entropy / max_entropy, 4) if max_entropy > 0 else 0.0


def _ignored_domain_features(
    metadata: dict[str, Any],
    ranked_features: list[str],
    ranked_importance: list[float],
    pca_used: bool,
) -> list[str]:
    if pca_used:
        return []
    total = sum(ranked_importance) or 1e-9
    low_importance_features = {
        f for f, v in zip(ranked_features, ranked_importance)
        if (v / total) < IGNORED_FEATURE_IMPORTANCE_RATIO
    }
    domain_columns = metadata.get("numerical_columns", []) + metadata.get("categorical_columns", [])
    ignored: list[str] = []
    for col in domain_columns:
        matched = [f for f in ranked_features if col in f]
        if not matched or all(f in low_importance_features for f in matched):
            ignored.append(col)
    return ignored


def build_explainability_summary(shap_result: dict[str, Any], state: PipelineState) -> dict[str, Any]:
    ranked_features = shap_result.get("ranked_features", [])
    ranked_importance = shap_result.get("ranked_importance", [])
    metadata = state.get("metadata", {})
    test_metrics = state.get("metrics", {}).get("best_model_metrics", {})
    problem_type = metadata.get("problem_type")
    pca_used = shap_result.get("pca_used", False)

    dominant_ratio = _dominant_feature_ratio(ranked_importance)
    diversity = _feature_diversity_score(ranked_importance)
    ignored = _ignored_domain_features(metadata, ranked_features, ranked_importance, pca_used)

    near_perfect = (
        (test_metrics.get("f1", 0.0) > 0.98 or test_metrics.get("roc_auc", 0.0) > 0.995)
        if problem_type == "classification"
        else test_metrics.get("r2", 0.0) > 0.995
    )
    suspected_target_leakage = bool(near_perfect and dominant_ratio > 0.5)

    notes: list[str] = []
    if dominant_ratio > 0.6:
        notes.append(
            f"Single feature '{ranked_features[0]}' accounts for {dominant_ratio:.0%} of total "
            "SHAP importance — possible over-reliance on one feature."
        )
    if suspected_target_leakage:
        notes.append(
            f"Near-perfect held-out performance combined with a dominant single feature "
            f"('{ranked_features[0]}', {dominant_ratio:.0%} of importance) suggests possible target leakage."
        )
    if diversity < 0.4 and len(ranked_features) > 3:
        notes.append(
            f"Low feature diversity score ({diversity:.2f}) — model decisions are concentrated "
            "in a small subset of features."
        )
    if ignored:
        shown = ignored[:5]
        notes.append(
            f"{len(ignored)} domain feature(s) contribute negligible signal and may be "
            f"candidates for removal: {shown}{' ...' if len(ignored) > 5 else ''}."
        )
    if not notes:
        notes.append(
            "No explainability concerns detected — importance is reasonably distributed "
            "and no single-feature dominance was observed."
        )

    return {
        "top_features": ranked_features[:5],
        "feature_importance": {f: round(v, 4) for f, v in zip(ranked_features[:10], ranked_importance[:10])},
        "dominant_feature_ratio": dominant_ratio,
        "suspected_target_leakage": suspected_target_leakage,
        "feature_diversity_score": diversity,
        "ignored_domain_features": ignored,
        "critic_explainability_notes": notes,
    }


def shap_explainability_node(state: PipelineState) -> PipelineState:
    try:
        raw_result = run_shap_explainability(state, generate_plots=False)
        state["explainability"] = build_explainability_summary(raw_result, state)
        state.setdefault("history", []).append({"node": "shap_explainability", "status": "ok"})
    except Exception as exc:  # noqa: BLE001
        state["explainability"] = {
            "top_features": [],
            "feature_importance": {},
            "dominant_feature_ratio": 0.0,
            "suspected_target_leakage": False,
            "feature_diversity_score": 0.0,
            "ignored_domain_features": [],
            "critic_explainability_notes": [f"SHAP explainability failed: {exc}"],
        }
        state.setdefault("history", []).append(
            {"node": "shap_explainability", "status": "error", "error": str(exc)}
        )
    return state