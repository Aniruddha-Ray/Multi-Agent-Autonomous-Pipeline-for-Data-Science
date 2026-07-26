"""Experience model: the persisted record of one completed pipeline run.

Extracted verbatim from Notebook Cell 13
("NEW CELL — EXPERIENCE SCORING: EXPERIENCE MODEL"). Contains the
``Experience`` dataclass, the ``build_experience`` assembler, the versioned
payload schema constant, and ``build_experience_payload`` — the function
that turns an ``Experience`` (plus a separately-computed score) into the
JSON-serializable dict persisted by the memory subsystem.

See Implementation Tasks 1/2 (persistence audit) and Task 5/6 (architecture
audit) in the original notebook commentary for the full rationale behind
the explicit preprocessing-provenance fields and the payload's shape.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional
import sys
import os
from pathlib import Path 

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.models.state import PipelineState


@dataclass
class Experience:
    """One completed pipeline run, assembled entirely from existing state —
    no new agent output is required to build this.

    Preprocessing provenance (Task 5/6, architecture audit) is preserved
    explicitly via four fields rather than folded silently into
    ``pipeline_configuration``: what the Planner recommended
    (``planner_preprocessing_decision``), what was actually executed
    (``executed_preprocessing``), a human-readable summary of the
    transformation (``transformation_summary``), and the training
    configuration used to produce the reported metrics
    (``training_configuration``). ``explainability_summary`` (Task 6) carries
    forward the compact SHAP-derived ExplainabilitySummary built by
    shap_explainability_node — never raw SHAP tensors, only the same
    top_features/feature_importance/dominant_feature_ratio/etc. dict the
    Critic reasoned over. None of these are left empty/None when the
    corresponding upstream state exists.
    """
    dataset_summary: dict[str, Any]
    pipeline_configuration: dict[str, Any]   # {preprocessing, feature_engineering, model}
    hyperparameters: dict[str, Any]
    evaluation_metrics: dict[str, Any]
    cv_metrics: dict[str, Any]
    critic_feedback: dict[str, Any]
    planner_reasoning: dict[str, Any]
    execution_time_seconds: float
    timestamp: str
    planner_preprocessing_decision: dict[str, Any] = field(default_factory=dict)
    executed_preprocessing: dict[str, Any] = field(default_factory=dict)
    transformation_summary: str = ""
    training_configuration: dict[str, Any] = field(default_factory=dict)
    explainability_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_experience(state: PipelineState, execution_time_seconds: float) -> Experience:
    """Assembled purely from existing PipelineState sub-dicts — reuses
    Stage B's dataset_summary, and CELL 6c/6d/6f/6g/6i/10 outputs unchanged.

    Field mapping matches the *actual* shapes produced by CELL 6f (Training)
    and CELL 6d/16 (Model Recommendation) — chosen_model and held-out metrics
    live under ``state["metrics"]`` (best_model_name / best_model_metrics),
    not under ``state["model_recommendation"]``.

    Preprocessing provenance (Task 5): the Planner's own preprocessing
    fields, the Feature Engineering Agent's *executed* decision (which, per
    Task 1, is Planner-authoritative), a readable transformation summary,
    and the training configuration (best hyperparameters / CV score / trial
    count, sourced from the Training Agent's ``tuning_results``) are all
    captured explicitly so nothing is left empty or None when the data
    exists in state.

    Explainability provenance (Task 6): ``state["explainability"]`` (built by
    shap_explainability_node, Cell 10) is carried through unchanged — the
    same compact summary the Critic reasoned over, so an Experience record
    fully reconstructs the reasoning chain: planner decision -> executed
    preprocessing -> executed transformations -> training configuration ->
    evaluation metrics -> explainability summary -> critic observations ->
    experience score.
    """
    metrics = state.get("metrics", {})
    planner_decision = state.get("planner_decision", {}) or {}
    fe_decision = state.get("features", {}).get("decision", {}) or {}
    models_state = state.get("models", {}) or {}
    tuning_results = models_state.get("tuning_results", {}) or {}
    best_model_name = metrics.get("best_model_name")
    best_tuning = tuning_results.get(best_model_name, {}) if best_model_name else {}

    planner_preprocessing_decision = {
        k: planner_decision[k]
        for k in (
            "imputation_strategy", "categorical_encoding", "scaling",
            "use_pca", "handle_imbalance", "feature_selection_k",
        )
        if k in planner_decision
    }
    executed_preprocessing = dict(fe_decision) if fe_decision else {}
    transformation_summary = (
        fe_decision.get("reasoning")
        or "No feature-engineering decision was recorded for this run."
    )
    training_configuration = {
        "best_model": best_model_name,
        "best_hyperparameters": best_tuning.get("best_params", {}),
        "cv_best_score": best_tuning.get("best_score"),
        "n_optuna_trials": best_tuning.get("n_trials"),
        "train_test_split": models_state.get("train_test_split", {}),
    }
    explainability_summary = dict(state.get("explainability", {}))

    return Experience(
        dataset_summary=state.get("dataset_summary", {}),          # Stage B
        pipeline_configuration={
            "preprocessing": executed_preprocessing or planner_preprocessing_decision,
            "feature_engineering": fe_decision,
            "model": best_model_name,
        },
        hyperparameters=training_configuration["best_hyperparameters"],
        evaluation_metrics=metrics.get("best_model_metrics", {}),
        cv_metrics={
            "cv_best_score": best_tuning.get("best_score"),
            "n_trials": best_tuning.get("n_trials"),
        },
        critic_feedback=state.get("critic", {}),
        planner_reasoning=planner_decision,
        execution_time_seconds=execution_time_seconds,
        timestamp=datetime.utcnow().isoformat(),
        planner_preprocessing_decision=planner_preprocessing_decision,
        executed_preprocessing=executed_preprocessing,
        transformation_summary=transformation_summary,
        training_configuration=training_configuration,
        explainability_summary=explainability_summary,
    )


# Payload schema version. Bump this whenever the shape of
# build_experience_payload()'s dict changes, so retrieval code (Implementation
# Task 6) can tell which fields to expect on a given stored record.
EXPERIENCE_PAYLOAD_SCHEMA_VERSION: int = 2


def build_experience_payload(
    experience: dict[str, Any], experience_score: Optional[float] = None
) -> dict[str, Any]:
    """Implementation Task 1/2 (persistence audit): the versioned, structured
    payload persisted alongside the existing searchable columns.

    Rather than adding one new SQLite column per Experience field (Task 2),
    every piece of reasoning history the Experience already carries —
    planner decision, executed preprocessing, preprocessing summary,
    executed feature transformations, training configuration, evaluation
    metrics, explainability summary, critic observations, experience score —
    is assembled here into a single JSON-serializable dict and stored as one
    TEXT/JSON column (``experience_payload_json``). A future PostgreSQL
    migration only needs to change that one column's type to JSONB; no
    application code building or reading this dict needs to change.

    ``experience`` is the plain dict form already produced by
    ``Experience.to_dict()`` — this is exactly what ``experience_scorer_node``
    stores at ``state["current_experience"]`` — so no reconstruction of the
    ``Experience`` dataclass is needed and the dataclass itself is left
    completely unmodified (Implementation Task 2).

    ``experience_score`` is passed in separately because it is not a field
    on ``Experience`` itself — it's computed afterwards by
    ``score_experience`` and only known to ``memory_update_policy_node``,
    which is where this function is called from.
    """
    return {
        "schema_version": EXPERIENCE_PAYLOAD_SCHEMA_VERSION,
        "planner_decision": experience.get("planner_reasoning"),
        "executed_preprocessing": experience.get("executed_preprocessing"),
        "transformation_summary": experience.get("transformation_summary"),
        "training_configuration": experience.get("training_configuration"),
        "evaluation_metrics": experience.get("evaluation_metrics"),
        "explainability_summary": experience.get("explainability_summary"),
        "critic_observations": experience.get("critic_feedback"),
        "experience_score": experience_score,
    }