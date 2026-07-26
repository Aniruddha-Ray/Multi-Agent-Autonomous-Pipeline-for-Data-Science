"""Feature Engineering Agent.

Extracted verbatim from Notebook Cell 7 ("CELL 6c — FEATURE ENGINEERING AGENT").
``feature_engineering_node`` takes ``llm_client`` as an explicit parameter
(Stage E5 DI decision).
"""
from __future__ import annotations

import json
from typing import Any, Optional
import sys
import os
from pathlib import Path 

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.agents.eda import _top_correlated_pairs
from src.models.agent_io import FeatureEngineeringDecision
from src.models.state import PipelineState


def _mock_feature_engineering_decision(
    metadata: dict[str, Any],
    planner_decision: Optional[dict[str, Any]] = None,
) -> FeatureEngineeringDecision:
    planner_decision = planner_decision or {}

    numerical_columns = metadata["numerical_columns"]
    categorical_columns = metadata["categorical_columns"]
    high_cardinality_columns = metadata["high_cardinality_columns"]
    low_cardinality_columns = [c for c in categorical_columns if c not in high_cardinality_columns]

    if numerical_columns:
        imputation_heuristic = "median"
    elif categorical_columns:
        imputation_heuristic = "most_frequent"
    else:
        imputation_heuristic = "none"

    if high_cardinality_columns and low_cardinality_columns:
        encoding_heuristic = "mixed"
    elif high_cardinality_columns:
        encoding_heuristic = "ordinal"
    else:
        encoding_heuristic = "onehot"

    scaling_heuristic = bool(numerical_columns)

    top_pairs = _top_correlated_pairs(metadata, top_k=1)
    strong_multicollinearity = bool(top_pairs) and abs(top_pairs[0][2]) > 0.9
    use_pca_heuristic = len(numerical_columns) > 15 or strong_multicollinearity

    estimated_post_encoding_width = (
        len(numerical_columns) + len(low_cardinality_columns) * 3 + len(high_cardinality_columns)
    )
    feature_selection_k_heuristic = (
        min(20, estimated_post_encoding_width) if estimated_post_encoding_width > 25 else None
    )

    handle_imbalance_heuristic = bool(metadata.get("is_imbalanced", False))

    imputation_strategy = planner_decision.get("imputation_strategy") or imputation_heuristic
    encoding_strategy = planner_decision.get("categorical_encoding") or encoding_heuristic
    scaling = (
        planner_decision["scaling"] != "none"
        if "scaling" in planner_decision
        else scaling_heuristic
    )
    use_pca = (
        planner_decision["use_pca"] if "use_pca" in planner_decision else use_pca_heuristic
    )
    pca_n_components = min(10, len(numerical_columns)) if use_pca and numerical_columns else None
    feature_selection_k = planner_decision.get("feature_selection_k")
    if feature_selection_k is None:
        feature_selection_k = feature_selection_k_heuristic
    handle_imbalance = (
        planner_decision["handle_imbalance"]
        if "handle_imbalance" in planner_decision
        else handle_imbalance_heuristic
    )

    imputation_source = "the Planner recommendation" if planner_decision.get("imputation_strategy") else (
        f"column-type mix and {metadata['pct_missing']:.2%} overall missingness"
    )
    source_note = (
        "Planner decision supplied — used as the primary execution plan; "
        "metadata heuristics applied only where the Planner was silent."
        if planner_decision
        else "No Planner decision available (standalone run) — falling back "
             "to metadata-derived heuristics for every field."
    )
    reasoning_parts = [
        source_note,
        f"{len(numerical_columns)} numerical / {len(categorical_columns)} categorical columns detected.",
        f"Imputation='{imputation_strategy}' chosen based on {imputation_source}.",
        f"Encoding='{encoding_strategy}': low-cardinality categoricals ({low_cardinality_columns or 'none'}) "
        f"get one-hot, high-cardinality columns ({high_cardinality_columns or 'none'}) get ordinal encoding "
        f"to avoid dimensionality blow-up.",
        f"Scaling={'enabled' if scaling else 'disabled'} "
        f"({'per Planner' if 'scaling' in planner_decision else 'per heuristic'}).",
    ]
    if use_pca:
        reasoning_parts.append(
            f"PCA enabled (n_components={pca_n_components}) "
            f"({'per Planner' if 'use_pca' in planner_decision else 'per heuristic'})."
        )
    if feature_selection_k:
        reasoning_parts.append(
            f"SelectKBest(k={feature_selection_k}) enabled "
            f"({'per Planner' if planner_decision.get('feature_selection_k') else 'per heuristic'})."
        )
    if handle_imbalance:
        reasoning_parts.append(
            f"Imbalance handling enabled (class_weight='balanced' where supported) "
            f"({'per Planner' if 'handle_imbalance' in planner_decision else 'per heuristic'})."
        )

    return FeatureEngineeringDecision(
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        high_cardinality_columns=high_cardinality_columns,
        imputation_strategy=imputation_strategy,
        encoding_strategy=encoding_strategy,
        scaling=scaling,
        use_pca=use_pca,
        pca_n_components=pca_n_components,
        feature_selection_k=feature_selection_k,
        handle_imbalance=handle_imbalance,
        reasoning=" ".join(reasoning_parts),
    )


def build_preprocessing_pipeline(decision: FeatureEngineeringDecision) -> ColumnTransformer:
    transformers: list[tuple[str, Pipeline, list[str]]] = []

    numeric_strategy = decision.imputation_strategy
    if numeric_strategy not in ("mean", "median", "most_frequent"):
        numeric_strategy = "median"

    if decision.numerical_columns:
        num_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy=numeric_strategy))]
        if decision.scaling:
            num_steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(num_steps), decision.numerical_columns))

    low_card_cols = [c for c in decision.categorical_columns if c not in decision.high_cardinality_columns]
    high_card_cols = decision.high_cardinality_columns

    if low_card_cols and decision.encoding_strategy in ("onehot", "mixed"):
        transformers.append((
            "cat_low",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            low_card_cols,
        ))

    if high_card_cols:
        high_card_encoder = (
            OneHotEncoder(handle_unknown="ignore")
            if decision.encoding_strategy == "onehot"
            else OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        )
        transformers.append((
            "cat_high",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", high_card_encoder),
            ]),
            high_card_cols,
        ))

    if not transformers:
        raise ValueError("Feature Engineering Agent found no usable columns to build a pipeline from.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def feature_engineering_node(state: PipelineState, llm_client: Any) -> PipelineState:
    metadata = state["metadata"]
    planner_decision = state.get("planner_decision")

    system_prompt = (
        "You are the Feature Engineering Agent in a multi-agent ML pipeline. "
        "The Planner has already decided a first-pass imputation, encoding, "
        "scaling, PCA and imbalance-handling strategy — treat it as the "
        "primary execution plan and only fill in gaps using dataset "
        "metadata. Return structured JSON."
    )
    user_prompt = json.dumps(
        {
            "metadata": {k: v for k, v in metadata.items() if k != "summary_statistics"},
            "planner_decision": planner_decision,
        },
        default=str,
    )
    decision = llm_client.structured_call(
        schema=FeatureEngineeringDecision,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        mock_fn=lambda: _mock_feature_engineering_decision(metadata, planner_decision),
    )

    preprocessor = build_preprocessing_pipeline(decision)

    state["features"] = {
        "decision": decision.model_dump(),
        "preprocessor": preprocessor,
    }
    state.setdefault("history", []).append({"node": "feature_engineering_agent", "status": "ok"})
    return state