"""Structured JSON schemas (pydantic) returned by each agent.

Extracted verbatim from Notebook Cell 5 ("CELL 5 — LANGGRAPH STATE") — the
five pydantic ``BaseModel`` schemas only. ``PipelineState`` lives in
``models/state.py`` (see that module's docstring for the split rationale).
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class PlannerDecision(BaseModel):
    """Structured decision produced by the Planner agent."""
    problem_type: Literal["classification", "regression"] = Field(...)
    imputation_strategy: Literal["mean", "median", "most_frequent", "none"] = Field(...)
    categorical_encoding: Literal["onehot", "ordinal", "mixed"] = Field(...)
    scaling: Literal["standard", "none"] = Field(...)
    use_pca: bool = Field(...)
    handle_imbalance: bool = Field(...)
    feature_selection_k: Optional[int] = Field(
        default=None,
        description="Planner's recommended SelectKBest width, or None if no "
                     "feature selection is recommended. Consumed by the "
                     "Feature Engineering Agent as the primary value, with "
                     "the Agent's own metadata-based estimate used only as "
                     "a fallback when this is None.",
    )
    candidate_models: list[str] = Field(...)
    reasoning: str = Field(...)


class EDAObservations(BaseModel):
    """Structured observations produced by the EDA agent."""
    key_observations: list[str]
    target_balance_note: str
    correlation_note: str
    artifact_paths: list[str]


class FeatureEngineeringDecision(BaseModel):
    """Structured decision produced by the Feature Engineering agent."""
    numerical_columns: list[str]
    categorical_columns: list[str]
    high_cardinality_columns: list[str]
    imputation_strategy: str
    encoding_strategy: str
    scaling: bool
    use_pca: bool
    pca_n_components: Optional[int] = None
    feature_selection_k: Optional[int] = None
    handle_imbalance: bool = False
    reasoning: str


class ModelRecommendation(BaseModel):
    """Structured decision produced by the Model Recommendation agent."""
    ranked_models: list[str]
    reasoning: dict[str, str]


class CriticReview(BaseModel):
    """Structured review produced by the Critic agent."""
    overfitting_detected: bool
    leakage_suspected: bool
    feature_engineering_ok: bool
    metrics_acceptable: bool
    issues: list[str] = Field(default_factory=list)
    recommendation: Literal["approve", "revise"]
    comments: str