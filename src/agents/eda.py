"""EDA Agent.

Extracted verbatim from Notebook Cell 6 ("CELL 6b — EDA AGENT").

``eda_agent_node`` takes ``llm_client`` as an explicit parameter (Stage E5
decision — mock calls preserved exactly; Stage F will inject the real
``LLMClient`` here without any further change to this function).
"""
from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import sys
import os
from pathlib import Path 

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.config.settings import Config
from src.models.agent_io import EDAObservations
from src.models.state import PipelineState


def _plot_histograms(df: pd.DataFrame, numerical_columns: list[str], artifacts_dir: str) -> str:
    cols = numerical_columns or []
    n = len(cols)
    if n == 0:
        return ""
    n_cols = min(3, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.2 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    for ax, col in zip(axes, cols):
        df[col].dropna().hist(ax=ax, bins=30, color="#4C72B0", edgecolor="white")
        ax.set_title(col, fontsize=10)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Feature Histograms", fontsize=13)
    fig.tight_layout()
    path = os.path.join(artifacts_dir, "eda_histograms.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _plot_boxplots(df: pd.DataFrame, numerical_columns: list[str], artifacts_dir: str) -> str:
    cols = numerical_columns or []
    n = len(cols)
    if n == 0:
        return ""
    n_cols = min(3, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.2 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    for ax, col in zip(axes, cols):
        ax.boxplot(df[col].dropna(), vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#DD8452", alpha=0.7))
        ax.set_title(col, fontsize=10)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Feature Boxplots (Outlier Inspection)", fontsize=13)
    fig.tight_layout()
    path = os.path.join(artifacts_dir, "eda_boxplots.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _plot_correlation_heatmap(corr_df: pd.DataFrame, artifacts_dir: str) -> str:
    if corr_df.empty or corr_df.shape[1] < 2:
        return ""
    fig, ax = plt.subplots(figsize=(max(5, 0.6 * corr_df.shape[1]), max(4, 0.6 * corr_df.shape[0])))
    im = ax.imshow(corr_df.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_xticklabels(corr_df.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr_df.index)))
    ax.set_yticklabels(corr_df.index, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation Heatmap", fontsize=13)
    fig.tight_layout()
    path = os.path.join(artifacts_dir, "eda_correlation_heatmap.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _plot_missing_values(df: pd.DataFrame, artifacts_dir: str) -> str:
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        return ""
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(missing)), 4))
    ax.bar(missing.index.astype(str), missing.values, color="#C44E52")
    ax.set_ylabel("Missing count")
    ax.set_title("Missing Values per Column", fontsize=13)
    ax.tick_params(axis="x", rotation=75, labelsize=8)
    fig.tight_layout()
    path = os.path.join(artifacts_dir, "eda_missing_values.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _plot_target_distribution(
    df: pd.DataFrame, target_column: str, problem_type: str, artifacts_dir: str
) -> str:
    if problem_type == "classification":
        counts = df[target_column].value_counts(dropna=False)
        fig = px.bar(
            x=counts.index.astype(str), y=counts.values,
            labels={"x": target_column, "y": "count"},
            title="Target Class Distribution",
        )
    else:
        fig = px.histogram(df, x=target_column, nbins=40, title="Target Distribution")
    path = os.path.join(artifacts_dir, "eda_target_distribution.html")
    fig.write_html(path)
    return path


def generate_eda_artifacts(df: pd.DataFrame, metadata: dict[str, Any], cfg: Config) -> dict[str, str]:
    corr_df = pd.DataFrame(metadata["correlation_matrix"])
    artifacts = {
        "histograms": _plot_histograms(df, metadata["numerical_columns"], cfg.artifacts_dir),
        "boxplots": _plot_boxplots(df, metadata["numerical_columns"], cfg.artifacts_dir),
        "correlation_heatmap": _plot_correlation_heatmap(corr_df, cfg.artifacts_dir),
        "missing_values": _plot_missing_values(df, cfg.artifacts_dir),
        "target_distribution": _plot_target_distribution(
            df, metadata["target_column"], metadata["problem_type"], cfg.artifacts_dir
        ),
    }
    return {name: path for name, path in artifacts.items() if path}


def _top_correlated_pairs(metadata: dict[str, Any], top_k: int = 3) -> list[tuple[str, str, float]]:
    """Extract the top-k most (absolute) correlated numerical feature pairs.

    Kept in the EDA domain per your Stage E4 Issue B decision — NOT moved to
    ``utils/`` and NOT duplicated. Imported directly from here by
    ``agents/feature_engineering.py`` and ``memory/dataset_summary.py``.
    """
    corr_df = pd.DataFrame(metadata["correlation_matrix"])
    if corr_df.empty or corr_df.shape[1] < 2:
        return []
    pairs: list[tuple[str, str, float]] = []
    cols = corr_df.columns.tolist()
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            val = corr_df.loc[c1, c2]
            if pd.notna(val):
                pairs.append((c1, c2, float(val)))
    pairs.sort(key=lambda t: abs(t[2]), reverse=True)
    return pairs[:top_k]


def _mock_eda_observations(metadata: dict[str, Any], artifact_paths: dict[str, str]) -> EDAObservations:
    observations: list[str] = [
        f"Dataset has {metadata['n_rows']} rows and {metadata['n_cols']} feature columns "
        f"({metadata['n_numerical']} numerical, {metadata['n_categorical']} categorical).",
        f"Overall missingness is {metadata['pct_missing']:.2%}.",
    ]
    if metadata["high_cardinality_columns"]:
        observations.append(
            f"High-cardinality categorical column(s) detected: {metadata['high_cardinality_columns']} "
            f"— candidates for target/frequency encoding rather than one-hot."
        )
    top_pairs = _top_correlated_pairs(metadata)
    if top_pairs:
        strongest = top_pairs[0]
        observations.append(
            f"Strongest numerical correlation is between '{strongest[0]}' and '{strongest[1]}' (r={strongest[2]:.2f})."
        )

    if metadata["problem_type"] == "classification":
        target_balance_note = (
            f"Target is imbalanced (majority:minority ratio = {metadata['imbalance_ratio']:.2f})."
            if metadata["is_imbalanced"]
            else f"Target classes are reasonably balanced (ratio = {metadata['imbalance_ratio']:.2f})."
        )
    else:
        target_balance_note = "Target is continuous; class-imbalance handling is not applicable."

    correlation_note = (
        f"Top correlated numerical pairs: {[(a, b, round(v, 2)) for a, b, v in top_pairs]}."
        if top_pairs
        else "No strong pairwise numerical correlations detected (or fewer than 2 numerical columns)."
    )

    return EDAObservations(
        key_observations=observations,
        target_balance_note=target_balance_note,
        correlation_note=correlation_note,
        artifact_paths=list(artifact_paths.values()),
    )


def eda_agent_node(state: PipelineState, llm_client: Any, cfg: Config) -> PipelineState:
    """LangGraph node: generate EDA plots and structured observations.

    ``llm_client``/``cfg`` are explicit parameters (Stage E5 DI decision) —
    the notebook read both as module globals (``llm_client``, ``CFG``).
    """
    import json as _json

    df = state["dataset"]
    metadata = state["metadata"]
    artifact_paths = generate_eda_artifacts(df, metadata, cfg)

    system_prompt = (
        "You are the EDA Agent in a multi-agent ML pipeline. Given dataset "
        "metadata, return structured exploratory observations as JSON."
    )
    user_prompt = _json.dumps(
        {k: v for k, v in metadata.items() if k != "summary_statistics"}, default=str
    )
    eda_observations = llm_client.structured_call(
        schema=EDAObservations,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        mock_fn=lambda: _mock_eda_observations(metadata, artifact_paths),
    )

    state["eda"] = {
        "observations": eda_observations.model_dump(),
        "artifact_paths": artifact_paths,
    }
    state.setdefault("history", []).append({"node": "eda_agent", "status": "ok"})
    return state