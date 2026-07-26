"""IPython display helpers for interactive (notebook/Jupyter) rendering.

Extracted verbatim from Notebook Cell 28 ("CELL 9 — DISPLAY OUTPUTS")
— all display_* functions and display_pipeline_outputs — plus
``display_explainability_section`` from Notebook Cell 25
("CELL 10 — SHAP EXPLAINABILITY"), placed here per the Stage E5.8
deferral: it is presentation-layer only (IPython display + plot
generation), unlike shap_explainability_node (agents/explainability.py),
which computes the compact state["explainability"] summary with no plots.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from IPython.display import HTML, IFrame, Image, Markdown, display
import sys
import os
from pathlib import Path 

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.agents.explainability import run_shap_explainability
from src.models.state import PipelineState


def display_dataset_summary(state: PipelineState) -> None:
    """Render the Dataset Analyzer's metadata as a compact summary table."""
    metadata = state["metadata"]
    display(Markdown("## 1. Dataset Summary"))
    display(Markdown(
        f"**Source:** {state['dataset_source']}  \n"
        f"**Target column:** `{metadata['target_column']}`  \n"
        f"**Problem type:** `{metadata['problem_type']}`"
    ))

    summary_rows = {
        "Rows": metadata["n_rows"],
        "Feature columns": metadata["n_cols"],
        "Numerical columns": metadata["n_numerical"],
        "Categorical columns": metadata["n_categorical"],
        "High-cardinality columns": len(metadata["high_cardinality_columns"]) or "none",
        "Overall missingness": f"{metadata['pct_missing']:.2%}",
        "Class imbalance ratio": f"{metadata['imbalance_ratio']:.2f}" if metadata["problem_type"] == "classification" else "n/a",
        "Imbalanced?": metadata.get("is_imbalanced", False) if metadata["problem_type"] == "classification" else "n/a",
    }
    summary_df = pd.DataFrame(summary_rows.items(), columns=["Metric", "Value"])
    display(summary_df)

    if metadata["high_cardinality_columns"]:
        display(Markdown(f"**High-cardinality columns:** {metadata['high_cardinality_columns']}"))


def display_eda_section(state: PipelineState) -> None:
    """Render the EDA Agent's charts and structured observations."""
    eda = state["eda"]
    display(Markdown("## 2. Exploratory Data Analysis"))

    for name, path in eda["artifact_paths"].items():
        if not path or not os.path.exists(path):
            continue
        title = name.replace("_", " ").title()
        display(Markdown(f"**{title}**"))
        if path.endswith(".html"):
            display(IFrame(src=path, width=760, height=460))
        else:
            display(Image(filename=path))

    observations = eda["observations"]
    display(Markdown("### Agent Observations"))
    for obs in observations["key_observations"]:
        display(Markdown(f"- {obs}"))
    display(Markdown(
        f"- **Target balance:** {observations['target_balance_note']}\n"
        f"- **Correlation note:** {observations['correlation_note']}"
    ))


def display_feature_engineering_section(state: PipelineState) -> None:
    """Render the Feature Engineering Agent's decision and reasoning."""
    decision = state["features"]["decision"]
    display(Markdown("## 3. Feature Engineering Decision"))

    fe_rows = {
        "Imputation strategy": decision["imputation_strategy"],
        "Encoding strategy": decision["encoding_strategy"],
        "Scaling applied": decision["scaling"],
        "PCA used": decision["use_pca"],
        "PCA components": decision["pca_n_components"] or "n/a",
        "Feature selection k": decision["feature_selection_k"] or "n/a",
        "Imbalance handling": decision.get("handle_imbalance", False),
    }
    display(pd.DataFrame(fe_rows.items(), columns=["Setting", "Value"]))
    display(Markdown(f"**Reasoning:** {decision['reasoning']}"))


def display_model_recommendation_section(state: PipelineState) -> None:
    """Render the Model Recommendation Agent's ranked shortlist and rationale."""
    recommendation = state["model_recommendation"]
    display(Markdown("## 4. Model Recommendation"))

    rec_df = pd.DataFrame({
        "Rank": range(1, len(recommendation["ranked_models"]) + 1),
        "Model": recommendation["ranked_models"],
        "Reasoning": [recommendation["reasoning"][m] for m in recommendation["ranked_models"]],
    })
    display(rec_df)


def plot_model_comparison(state: PipelineState) -> None:
    """Bar chart (matplotlib) + interactive grouped bar (Plotly) comparing
    every trained candidate's held-out metrics, with the best model highlighted."""
    metrics = state["metrics"]
    per_model = metrics["per_model_test_metrics"]
    best_model_name = metrics["best_model_name"]
    problem_type = state["metadata"]["problem_type"]

    metrics_df = pd.DataFrame(per_model).T
    metrics_df.index.name = "model"

    display(Markdown("## 5. Model Training Results"))
    display(Markdown(
        f"**Best model:** `{best_model_name}` "
        f"(primary scorer: `{metrics['primary_scorer']}`)"
    ))
    display(metrics_df.round(4))

    primary_metric = "f1" if problem_type == "classification" else "rmse"
    ascending_is_better = primary_metric == "rmse"
    values = metrics_df[primary_metric]

    fig, ax = plt.subplots(figsize=(max(5, 1.3 * len(values)), 4))
    colors = [
        "#55A868" if model_name == best_model_name else "#4C72B0"
        for model_name in values.index
    ]
    ax.bar(values.index.astype(str), values.values, color=colors)
    ax.set_ylabel(primary_metric.upper())
    ax.set_title(f"Held-out {primary_metric.upper()} by Model"
                 f"{' (lower is better)' if ascending_is_better else ' (higher is better)'}")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    plt.show()

    fig_px = px.bar(
        metrics_df.reset_index(), x="model", y=primary_metric,
        color=metrics_df.reset_index()["model"].eq(best_model_name).map(
            {True: "Best model", False: "Candidate"}
        ),
        title=f"Interactive comparison — held-out {primary_metric.upper()}",
        labels={"color": ""},
    )
    fig_px.show()


def display_explainability_section(state: PipelineState) -> None:
    """Render the SHAP summary/bar plots and observations, in the same style
    as the other display_* functions. Presentation-layer only — recomputes
    SHAP with plots for the report; does not write anything back into state."""
    display(Markdown("## 8. Explainability (SHAP)"))
    try:
        full_result = run_shap_explainability(state, generate_plots=True)
    except Exception as exc:  # noqa: BLE001
        display(Markdown(f"_SHAP explainability could not be computed: {exc}_"))
        return

    display(Markdown(f"**Explainer:** `{full_result['explainer_type']}` on `{full_result['best_model_name']}`"))
    for path_key in ("summary_plot", "feature_importance_bar"):
        path = full_result["artifact_paths"].get(path_key)
        if path and os.path.exists(path):
            display(Image(filename=path))

    display(Markdown("### Observations"))
    for obs in full_result["observations"]:
        display(Markdown(f"- {obs}"))

    summary = state.get("explainability", {})
    if summary.get("critic_explainability_notes"):
        display(Markdown("### Critic-Facing Explainability Notes"))
        for note in summary["critic_explainability_notes"]:
            display(Markdown(f"- {note}"))


def display_critic_section(state: PipelineState) -> None:
    """Render the Critic Agent's review verdict and any flagged issues."""
    critic = state["critic"]
    display(Markdown("## 7. Critic Review"))
    display(Markdown(f"**Recommendation:** `{critic['recommendation'].upper()}`"))

    critic_rows = {
        "Overfitting detected": critic["overfitting_detected"],
        "Leakage suspected": critic["leakage_suspected"],
        "Feature engineering OK": critic["feature_engineering_ok"],
        "Metrics acceptable": critic["metrics_acceptable"],
    }
    display(pd.DataFrame(critic_rows.items(), columns=["Check", "Result"]))

    if critic["issues"]:
        display(Markdown("**Issues raised:**"))
        for issue in critic["issues"]:
            display(Markdown(f"- {issue}"))
    display(Markdown(f"**Comments:** {critic['comments']}"))


def display_memory_section(state: PipelineState) -> None:
    """Render what the Memory Retrieval Agent found before planning, and
    what the Experience Scorer / Memory Update Policy decided to do with
    this run's outcome afterward."""
    display(Markdown("## 8. Memory"))

    retrieved = state.get("retrieved_memories", [])
    display(Markdown("**Similar past runs retrieved before planning:**"))
    if retrieved:
        display(pd.DataFrame([
            {"run_id": m["run_id"], "similarity": m["similarity"], "usefulness": m["usefulness"],
             "chosen_model": m.get("chosen_model"), "quality_label": m["quality_label"]}
            for m in retrieved
        ]))
    else:
        display(Markdown("_No similar past runs were found in memory (expected on the first run)._"))

    scored = state.get("experience_score")
    decision = state.get("memory_update_decision")
    if scored and decision:
        display(Markdown(
            f"**This run's experience score:** {scored['experience_score']:.3f} "
            f"(recommendation: `{scored['recommendation']}`)  \n"
            f"**Memory update decision:** `{decision['action']}` — {decision['reason']}  \n"
            f"**Total runs stored:** {decision.get('total_runs_stored', 'n/a')}"
        ))


def display_pipeline_outputs(state: PipelineState) -> None:
    """Render every section of the finished pipeline run, in actual graph
    execution order: dataset summary -> EDA -> feature engineering ->
    model recommendation -> training results -> explainability (SHAP) ->
    critic review -> memory (retrieval + experience/update)."""
    display_dataset_summary(state)
    display_eda_section(state)
    display_feature_engineering_section(state)
    display_model_recommendation_section(state)
    plot_model_comparison(state)
    display_explainability_section(state)
    display_critic_section(state)
    display_memory_section(state)