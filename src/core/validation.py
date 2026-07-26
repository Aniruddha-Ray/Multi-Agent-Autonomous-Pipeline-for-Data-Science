"""Pre-training dataset validation.

Extracted verbatim from Notebook Cell 5 ("CELL 6a...") — only the
``PipelineValidationError`` exception and ``validate_dataset_for_training``
function. Fails fast with an informative message before any agent reasons
over the dataset, instead of letting the same problem surface later as an
opaque library error deep inside training (Task 7, architecture audit).
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd
import sys
import os
from pathlib import Path 

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.settings import Config
from src.core.metadata import _infer_problem_type


class PipelineValidationError(ValueError):
    """Raised when the input dataset fails a pre-training sanity check.

    Carries a clear, human-readable message describing exactly which check
    failed, instead of letting the failure surface later as a raw/opaque
    library error deep inside the graph (e.g. a bare KeyError for a missing
    target column, or a StratifiedKFold ValueError with no dataset context).
    """


def validate_dataset_for_training(df: pd.DataFrame, target_column: str, cfg: Config) -> None:
    """Fail fast, with an informative message, before any agent reasons over
    the dataset. Called once from ``dataset_analyzer_node`` so every
    downstream agent (Planner, Feature Engineering, Training, ...) can
    assume these invariants already hold.
    """
    if df is None or df.empty:
        raise PipelineValidationError("Dataset is empty — nothing to analyze or train on.")

    if target_column not in df.columns:
        raise PipelineValidationError(
            f"Target column '{target_column}' was not found in the dataset. "
            f"Available columns: {list(df.columns)}"
        )

    feature_columns = [c for c in df.columns if c != target_column]
    if not feature_columns:
        raise PipelineValidationError(
            f"Dataset has no feature columns besides the target column '{target_column}'. "
            "At least one feature column is required to train a model."
        )

    target = df[target_column]
    if target.isna().all():
        raise PipelineValidationError(f"Target column '{target_column}' is entirely missing (all NaN).")

    n_valid_rows = int(target.notna().sum())
    min_rows_required = max(cfg.cv_folds, 10)
    if n_valid_rows < min_rows_required:
        raise PipelineValidationError(
            f"Only {n_valid_rows} row(s) have a non-null target value — at least "
            f"{min_rows_required} are required for a {cfg.cv_folds}-fold cross-validated "
            "training run. Reduce cv_folds or supply more data."
        )

    n_unique_target = int(target.nunique(dropna=True))
    if n_unique_target < 2:
        raise PipelineValidationError(
            f"Target column '{target_column}' has only {n_unique_target} distinct value(s) — "
            "there is nothing to learn (at least 2 distinct values are required)."
        )

    problem_type = _infer_problem_type(target)
    if problem_type == "classification":
        class_counts = target.value_counts(dropna=True)
        smallest_class_size = int(class_counts.min())
        if smallest_class_size < cfg.cv_folds:
            raise PipelineValidationError(
                f"The smallest class in target column '{target_column}' has only "
                f"{smallest_class_size} sample(s), which is fewer than cv_folds={cfg.cv_folds}. "
                "Stratified cross-validation is not feasible — reduce cv_folds in Config or "
                "collect more samples for that class."
            )
    else:
        if n_valid_rows < cfg.cv_folds:
            raise PipelineValidationError(
                f"Only {n_valid_rows} valid row(s) available, fewer than cv_folds={cfg.cv_folds} "
                "— K-fold cross-validation is not feasible. Reduce cv_folds or supply more data."
            )