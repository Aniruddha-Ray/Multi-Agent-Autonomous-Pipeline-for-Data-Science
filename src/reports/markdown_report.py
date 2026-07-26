"""Markdown report generator.

Extracted verbatim from Notebook Cell 29 ("CELL 11 — MARKDOWN REPORT GENERATOR").
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any
import sys
import os
from pathlib import Path 

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.agents.explainability import run_shap_explainability
from src.config.settings import Config
from src.models.state import PipelineState


def _format_metrics_table(metrics: dict[str, float]) -> str:
    """Render a flat metrics dict as a two-column Markdown table."""
    rows = "\n".join(f"| {name} | {value:.4f} |" for name, value in metrics.items())
    return f"| Metric | Value |\n|---|---|\n{rows}"


def _format_model_comparison_table(metrics: dict[str, Any]) -> str:
    """Render every trained candidate's held-out metrics as one Markdown table,
    bolding the best model's row."""
    per_model = metrics["per_model_test_metrics"]
    best_model_name = metrics["best_model_name"]
    metric_names = list(next(iter(per_model.values())).keys())

    header = "| Model | " + " | ".join(m.upper() for m in metric_names) + " |"
    divider = "|---|" + "---|" * len(metric_names)
    rows = []
    for model_name, model_metrics in per_model.items():
        values = " | ".join(f"{model_metrics[m]:.4f}" for m in metric_names)
        label = f"**{model_name}**" if model_name == best_model_name else model_name
        rows.append(f"| {label} | {values} |")
    return "\n".join([header, divider, *rows])


def _format_experience_section(state: PipelineState) -> str:
    scored = state.get("experience_score", {})
    decision = state.get("memory_update_decision", {})
    lines = [
        "## Experience & Memory",
        f"- **Experience score:** {scored.get('experience_score', 'n/a')}",
        f"- **Generalization score:** {scored.get('generalization_score', 'n/a')}",
        f"- **Confidence:** {scored.get('confidence', 'n/a')}",
        f"- **Scorer reasoning:** {scored.get('reasoning', 'n/a')}",
        "",
        f"- **Memory update action:** {decision.get('action', 'n/a')}",
        f"- **Stored?** {'Yes' if decision.get('action') in ('store_new', 'replace', 'merge') else 'No'}",
        f"- **Replaced existing memory?** {'Yes — run ' + str(decision.get('replaced_run_id')) if decision.get('action') == 'replace' else 'No'}",
        f"- **Reason:** {state.get('update_reason', 'n/a')}",
    ]
    similar = state.get("similar_memories", [])
    if similar:
        lines.append("\n**Retrieved similar memories considered:**")
        for m in similar:
            lines.append(f"- run {m['run_id']} (similarity={m['similarity']:.2f}, model={m.get('chosen_model')})")
    return "\n".join(lines)


def _generate_future_improvements(state: PipelineState, cfg: Config) -> list[str]:
    """Deterministic, rule-based suggestions for next iterations — derived
    from the same dataset/critic signals the Planner and Critic already
    reasoned over, rather than free-form LLM speculation.

    ``cfg`` is an explicit parameter here (the notebook read the module
    global ``CFG`` for ``n_optuna_trials``).
    """
    metadata = state["metadata"]
    critic = state["critic"]
    retrieved_memories = state.get("retrieved_memories", [])
    suggestions: list[str] = []

    if metadata["is_imbalanced"]:
        suggestions.append(
            "Address class imbalance explicitly (e.g. class weighting, SMOTE, "
            "or a threshold-tuned decision boundary) rather than relying on "
            "weighted averaging in the metrics alone."
        )
    if metadata["pct_missing"] > 0.05:
        suggestions.append(
            "Investigate the missingness mechanism (MCAR/MAR/MNAR) rather "
            "than defaulting to median/most-frequent imputation, since >5% "
            "of cells are missing overall."
        )
    if metadata["high_cardinality_columns"]:
        suggestions.append(
            "Revisit encoding for high-cardinality columns "
            f"({', '.join(metadata['high_cardinality_columns'])}) — target or "
            "frequency encoding may generalize better than the current strategy."
        )
    if critic["issues"]:
        suggestions.append(
            "Resolve the Critic's outstanding issues before treating this "
            "run's metrics as final: " + "; ".join(critic["issues"])
        )
    if not retrieved_memories:
        suggestions.append(
            "No similar past runs existed in memory for this dataset fingerprint — "
            "future runs on related datasets should retrieve and reuse this run's "
            "successful configuration."
        )
    suggestions.append(
        f"Expand the Optuna budget past {cfg.n_optuna_trials} trials on "
        f"`{state['models']['best_model_name']}` specifically, now that it is "
        "the identified best candidate rather than one of several unknowns."
    )
    return suggestions


def generate_markdown_report(state: PipelineState, cfg: Config) -> str:
    """Assemble the full run report as a single Markdown string: dataset
    summary, EDA, preprocessing, model, metrics, critic comments, memory
    usage, future improvements.
    """
    metadata = state["metadata"]
    eda = state["eda"]
    decision = state["features"]["decision"]
    recommendation = state["model_recommendation"]
    metrics = state["metrics"]
    critic = state["critic"]
    explainability_summary = state.get("explainability", {})

    sections: list[str] = []

    sections.append(
        f"# Pipeline Report — {state['dataset_source']}\n\n"
        f"_Generated {datetime.now().isoformat(timespec='seconds')}_"
    )

    sections.append(
        "## 1. Dataset Summary\n\n"
        f"- **Target column:** `{metadata['target_column']}`\n"
        f"- **Problem type:** `{metadata['problem_type']}`\n"
        f"- **Shape:** {metadata['n_rows']} rows × {metadata['n_cols']} feature columns "
        f"({metadata['n_numerical']} numerical, {metadata['n_categorical']} categorical)\n"
        f"- **Missingness:** {metadata['pct_missing']:.2%} overall\n"
        f"- **Class imbalance ratio:** "
        f"{metadata['imbalance_ratio']:.2f} (imbalanced={metadata['is_imbalanced']})"
        if metadata["problem_type"] == "classification"
        else f"- **Missingness:** {metadata['pct_missing']:.2%} overall"
    )

    observations = eda["observations"]
    sections.append(
        "## 2. Exploratory Data Analysis\n\n"
        + "\n".join(f"- {obs}" for obs in observations["key_observations"])
        + f"\n- **Target balance:** {observations['target_balance_note']}"
        + f"\n- **Correlation note:** {observations['correlation_note']}"
    )

    sections.append(
        "## 3. Feature Engineering\n\n"
        f"- **Imputation:** {decision['imputation_strategy']}\n"
        f"- **Encoding:** {decision['encoding_strategy']}\n"
        f"- **Scaling:** {decision['scaling']}\n"
        f"- **PCA:** {decision['use_pca']}"
        + (f" ({decision['pca_n_components']} components)" if decision["use_pca"] else "")
        + f"\n- **Feature selection k:** {decision['feature_selection_k'] or 'n/a'}"
        + f"\n- **Imbalance handling:** {decision.get('handle_imbalance', False)}\n\n"
        f"**Reasoning:** {decision['reasoning']}"
    )

    sections.append(
        "## 4. Model Selection\n\n"
        f"**Chosen model:** `{metrics['best_model_name']}`\n\n"
        "**Candidates considered, in ranked order:**\n\n"
        + "\n".join(
            f"{i}. **{m}** — {recommendation['reasoning'][m]}"
            for i, m in enumerate(recommendation["ranked_models"], start=1)
        )
    )

    sections.append(
        "## 5. Metrics\n\n"
        f"Primary scorer: `{metrics['primary_scorer']}`\n\n"
        "**Best model held-out metrics:**\n\n"
        f"{_format_metrics_table(metrics['best_model_metrics'])}\n\n"
        "**All candidates compared:**\n\n"
        f"{_format_model_comparison_table(metrics)}"
    )

    # The report is a presentation-layer consumer of PipelineState, not part
    # of the graph, so it recomputes SHAP with plots-off to get the full
    # ranked feature list for the write-up (state["explainability"] only
    # holds the compact ExplainabilitySummary the Critic reasoned over,
    # included below as "Critic-facing notes").
    try:
        shap_full = run_shap_explainability(state, generate_plots=False)
        top5 = list(zip(shap_full["ranked_features"][:5], shap_full["ranked_importance"][:5]))
        critic_notes = explainability_summary.get("critic_explainability_notes", [])
        sections.append(
            "## 6. Explainability (SHAP)\n\n"
            f"Explainer: `{shap_full['explainer_type']}`\n\n"
            "**Top 5 features by mean |SHAP|:**\n\n"
            + "\n".join(f"{i}. `{name}` — {value:.4f}" for i, (name, value) in enumerate(top5, start=1))
            + "\n\n"
            + "\n".join(f"- {obs}" for obs in shap_full["observations"])
            + ("\n\n**Critic-facing explainability notes:**\n\n" + "\n".join(f"- {n}" for n in critic_notes)
               if critic_notes else "")
        )
    except Exception as exc:
        sections.append(f"## 6. Explainability (SHAP)\n\n_Not available: {exc}_")

    sections.append(
        "## 7. Critic Review\n\n"
        f"**Recommendation:** `{critic['recommendation'].upper()}`\n\n"
        f"- Overfitting detected: {critic['overfitting_detected']}\n"
        f"- Leakage suspected: {critic['leakage_suspected']}\n"
        f"- Feature engineering OK: {critic['feature_engineering_ok']}\n"
        f"- Metrics acceptable: {critic['metrics_acceptable']}\n\n"
        + ("**Issues raised:**\n" + "\n".join(f"- {issue}" for issue in critic["issues"]) + "\n\n" if critic["issues"] else "")
        + f"**Comments:** {critic['comments']}"
    )

    sections.append(
        "## 8. Memory\n\n"
        + (
            "**Similar past runs retrieved before planning:**\n"
            + "\n".join(
                f"  - run {m['run_id']} (similarity={m['similarity']:.2f}, "
                f"usefulness={state.get('retrieval_scores', {}).get(m['run_id'], m.get('usefulness', 0.0)):.2f}, "
                f"model={m.get('chosen_model')})"
                for m in state.get("retrieved_memories", [])
            )
            if state.get("retrieved_memories")
            else "No similar past runs were found in memory (expected on the first run)."
        )
    )

    sections.append(
        "## 9. Future Improvements\n\n"
        + "\n".join(f"- {s}" for s in _generate_future_improvements(state, cfg))
    )

    sections.append(_format_experience_section(state))

    n_iterations = max((e.get("iteration", 0) for e in state.get("history", [])), default=0)
    sections.append(
        "## Appendix: Execution Trace\n\n"
        f"Total revision iterations: {state.get('iteration', 0)}\n\n"
        + "\n".join(
            f"{i}. `{e['node']}` — status={e['status']}"
            for i, e in enumerate(state.get("history", []), start=1)
        )
    )

    return "\n\n".join(sections) + "\n"


def report_generation_node(state: PipelineState, cfg: Config) -> PipelineState:
    """LangGraph-node-style helper: build the Markdown report and persist it
    to ``cfg.artifacts_dir/pipeline_report.md``.

    NOT wired into build_pipeline_graph (Stage E6) — matches the notebook's
    own design, where this is invoked manually after the graph reaches END,
    not as a graph node itself.
    """
    report_md = generate_markdown_report(state, cfg)
    report_path = os.path.join(cfg.artifacts_dir, "pipeline_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    state["report"] = report_md
    state.setdefault("history", []).append({"node": "report_generation", "status": "ok", "path": report_path})
    return state