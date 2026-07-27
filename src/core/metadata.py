"""Dataset metadata calculation.

Extracted verbatim from Notebook Cell 5 ("CELL 6a...") — only
``_infer_target_column``, ``_infer_problem_type``, and
``compute_dataset_metadata``. Pure Python/pandas; no LLM call.

NOTE (flagged during Stage E2 analysis, not silently fixed): this module
reads ``CFG.high_cardinality_threshold`` and ``CFG.imbalance_ratio_threshold``
directly from the module-level ``CFG`` singleton rather than taking a ``cfg``
parameter, exactly as the notebook does. This is inconsistent with
``core/validation.py``'s ``validate_dataset_for_training``, which *does* take
``cfg`` explicitly — that inconsistency already existed in the notebook and
is preserved here unchanged, per this project's "never fix logic, only
relocate it" rule.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path 

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.settings import CFG


def _infer_target_column(df: pd.DataFrame) -> str:
    """Pick the target column: prefer a column literally named 'target',
    otherwise fall back to the last column of the frame."""
    for candidate in ("target", "label", "y"):
        if candidate in df.columns:
            return candidate
    return df.columns[-1]

def resolve_target_column(df: pd.DataFrame) -> str:
    raw = input(f"Target column (columns: {list(df.columns)}) — press Enter to auto-detect: ").strip()
    if raw:
        if raw not in df.columns:
            raise ValueError(f"'{raw}' is not a column in the dataset.")
        return raw
    return _infer_target_column(df)


def _infer_problem_type(series: pd.Series) -> Literal["classification", "regression"]:
    """Heuristically decide classification vs. regression from the target."""
    if series.dtype == object or str(series.dtype).startswith("category"):
        return "classification"
    n_unique = series.nunique(dropna=True)
    if n_unique <= max(20, int(0.05 * len(series))):
        return "classification"
    return "regression"


def compute_dataset_metadata(df: pd.DataFrame, target_column: Optional[str] = None) -> dict[str, Any]:
    """Deterministically compute the dataset metadata block described in the
    master prompt: shape, missingness, column typing, target type, class
    imbalance, correlations and summary statistics.

    This function is pure Python/pandas — the Dataset Analyzer step does not
    require an LLM call, only downstream agents (Planner, EDA, Feature
    Engineering, ...) reason over its structured output.
    """
    target_column = target_column or _infer_target_column(df)
    feature_df = df.drop(columns=[target_column])

    numerical_columns = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [c for c in feature_df.columns if c not in numerical_columns]

    missing_per_column = df.isna().sum().to_dict()
    pct_missing = float(df.isna().mean().mean())

    cardinality = {c: int(df[c].nunique(dropna=True)) for c in categorical_columns}
    avg_cardinality = float(np.mean(list(cardinality.values()))) if cardinality else 0.0

    problem_type = _infer_problem_type(df[target_column])

    imbalance_ratio = 1.0
    class_counts: dict[str, int] = {}
    if problem_type == "classification":
        counts = df[target_column].value_counts(dropna=True)
        class_counts = {str(k): int(v) for k, v in counts.items()}
        if len(counts) >= 2:
            imbalance_ratio = float(counts.iloc[0] / max(counts.iloc[-1], 1))

    corr_matrix = feature_df[numerical_columns].corr(numeric_only=True) if numerical_columns else pd.DataFrame()
    summary_statistics = df.describe(include="all").to_dict()

    metadata: dict[str, Any] = {
        "target_column": target_column,
        "problem_type": problem_type,
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1] - 1),
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "n_numerical": len(numerical_columns),
        "n_categorical": len(categorical_columns),
        "missing_per_column": missing_per_column,
        "pct_missing": pct_missing,
        "cardinality": cardinality,
        "avg_cardinality": avg_cardinality,
        "high_cardinality_columns": [
            c for c, card in cardinality.items() if card > CFG.high_cardinality_threshold
        ],
        "class_counts": class_counts,
        "imbalance_ratio": imbalance_ratio,
        "is_imbalanced": imbalance_ratio > CFG.imbalance_ratio_threshold,
        "correlation_matrix": corr_matrix.round(4).to_dict(),
        "summary_statistics": summary_statistics,
    }
    return metadata