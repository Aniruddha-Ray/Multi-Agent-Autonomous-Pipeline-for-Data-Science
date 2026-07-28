"""Composition root for the Adaptive Multi-Agent Autonomous Data Science
Pipeline.

This is the ONE place in the package where every singleton object gets
constructed: Config, StructuredMemory, SemanticMemory, EmbeddingProvider,
MemoryRepository, and the LLM client (mock or real Anthropic backend,
auto-selected based on ANTHROPIC_API_KEY). No other module in src/
instantiates any of these (Stage E4/E5/E6 decisions) — every one of them
takes these objects as explicit constructor or function parameters instead.

Mirrors the notebook's own implicit construction order: Config -> Dataset
-> Memory (StructuredMemory -> SemanticMemory -> EmbeddingProvider ->
MemoryRepository) -> LLM client -> Graph -> run_pipeline -> display/report.
"""
from __future__ import annotations

from typing import Any
import sys
import os
from dataclasses import replace
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.config.settings import CFG, Config
from src.core.data_loader import load_train_test
from src.core.metadata import prompt_problem_type, resolve_target_column
from src.graph.build import build_pipeline_graph, run_pipeline
from src.llm.client import LLMClient
from src.memory.embeddings import get_embedding_provider
from src.memory.repository import MemoryRepository
from src.memory.semantic_store import SemanticMemory
from src.memory.structured_store_postgres import PostgresStructuredMemory
from src.memory.structured_store import StructuredMemory
from src.reports.display import display_pipeline_outputs
from src.reports.markdown_report import report_generation_node
from src.reports.run_trace import print_run_trace
from src.models.state import PipelineState  # add to the import block

import logging
from src.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

def build_dependencies(cfg: Config) -> dict[str, Any]:
    """Construct every singleton the graph needs, in the dependency order
    established across Stage E4 (memory subsystem) and Stage E6 (graph
    wiring): StructuredMemory -> SemanticMemory -> EmbeddingProvider ->
    MemoryRepository.
    """
    if cfg.memory_backend == "postgres":
        structured = PostgresStructuredMemory(cfg.postgres_dsn)
    else:
        structured = StructuredMemory(cfg.database_path)
    structured.run_startup_migrations()

    semantic = SemanticMemory(cfg.faiss_dim)
    embedding_provider = get_embedding_provider(cfg)
    memory_repository = MemoryRepository(structured, semantic, embedding_provider)

    llm_client = LLMClient(cfg)
    logger.info(f"LLMClient ready — backend: "
          f"{'real (' + cfg.llm_model + ')' if llm_client._real_llm_available else 'mock (deterministic heuristics)'}")

    return {
        "structured": structured,
        "semantic": semantic,
        "embedding_provider": embedding_provider,
        "memory_repository": memory_repository,
        "llm_client": llm_client,
    }

def get_runtime_config() -> Config:
    """Resolve the runtime Config exactly as the CLI does: mock LLM iff
    GROQ_API_KEY is absent. Single source of truth — reused by main() and
    by the FastAPI dependency layer so the two never diverge."""
    return replace(CFG, use_mock_llm=os.environ.get("GROQ_API_KEY") is None)


def run_end_to_end(
    df: pd.DataFrame, dataset_source: str, cfg: Config, deps: dict[str, Any],
    target_column: str | None = None, problem_type: str | None = None,
) -> PipelineState:
    """Execute one full run: build graph -> run_pipeline -> report
    generation. This is the exact sequence main() used to run inline,
    with the printing removed. FastAPI's /pipeline/run calls this same
    function — no second execution path exists anywhere."""
    graph = build_pipeline_graph(
        cfg=cfg,
        llm_client=deps["llm_client"],
        memory_repository=deps["memory_repository"],
        structured=deps["structured"],
        semantic=deps["semantic"],
    )
    final_state = run_pipeline(df, dataset_source, graph, cfg, target_column=target_column, problem_type=problem_type)
    final_state = report_generation_node(final_state, cfg)
    return final_state

def main() -> None:
    cfg = get_runtime_config()
    deps = build_dependencies(cfg)

    train_df, test_df, dataset_source, test_source = load_train_test(cfg)
    target_column = resolve_target_column(train_df)
    logger.info(f"Running pipeline graph on: {dataset_source}")
    logger.info(f"Dataset shape            : {train_df.shape}\n")

    problem_type = prompt_problem_type(train_df, target_column)
    final_state = run_end_to_end(train_df, dataset_source, cfg, deps, target_column=target_column, problem_type=problem_type)

    
    print_run_trace(final_state, cfg)

    logger.info("FINAL RESULT")
    
    logger.info(f"Problem type      : {final_state['metadata']['problem_type']}")
    logger.info(f"Best model        : {final_state['metrics']['best_model_name']}")
    logger.info(f"Best model metrics: { {k: round(v, 4) for k, v in final_state['metrics']['best_model_metrics'].items()} }")
    logger.info(f"Critic verdict    : {final_state['critic']['recommendation'].upper()}")
    logger.info(f"Critic comments   : {final_state['critic']['comments']}")
    logger.info(f"Experience score  : {final_state['experience_score']['experience_score']:.3f} "
          f"({final_state['experience_score']['recommendation']})")
    logger.info(f"Memory decision   : {final_state['memory_update_decision']['action']} "
          f"— {final_state['memory_update_decision']['reason']}")
    logger.info(f"Total runs stored : {final_state['memory_update_decision'].get('total_runs_stored', 'n/a')}")

    logger.info(f"\nReport written to: {cfg.artifacts_dir}/pipeline_report.md "
          f"({len(final_state['report'])} characters)")
    best_name = final_state["metrics"]["best_model_name"]
    best_pipeline = final_state["models"]["trained_pipelines"][best_name]
    test_features = test_df.drop(columns=[target_column], errors="ignore")
    preds = best_pipeline.predict(test_features)

    result_df = test_df.copy()
    result_df[target_column] = preds
    pred_path = f"{cfg.artifacts_dir}/prediction.csv"
    result_df.to_csv(pred_path, index=False)
    logger.info(f"Predictions written to: {pred_path} ({len(result_df)} rows)")

    try:
        get_ipython()  # type: ignore[name-defined]  # noqa: F821
        display_pipeline_outputs(final_state)
    except NameError:
        logger.info("\n(Skipping display_pipeline_outputs — not running inside IPython/Jupyter.)")


if __name__ == "__main__":
    main()