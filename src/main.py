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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.config.settings import CFG, Config
from src.core.data_loader import load_dataset
from src.graph.build import build_pipeline_graph, run_pipeline
from src.llm.client import LLMClient
from src.memory.embeddings import get_embedding_provider
from src.memory.repository import MemoryRepository
from src.memory.semantic_store import SemanticMemory
from src.memory.structured_store_postgres import PostgresStructuredMemory
from src.memory.structured_store import (
    StructuredMemory,
    # _ensure_experience_payload_column,
    # _ensure_memory_quality_columns,
)
from src.reports.display import display_pipeline_outputs
from src.reports.markdown_report import report_generation_node
from src.reports.run_trace import print_run_trace


def build_dependencies(cfg: Config) -> dict[str, Any]:
    """Construct every singleton the graph needs, in the dependency order
    established across Stage E4 (memory subsystem) and Stage E6 (graph
    wiring): StructuredMemory -> SemanticMemory -> EmbeddingProvider ->
    MemoryRepository.
    """
    if cfg.memory_backend == "postgres":
        structured = PostgresStructuredMemory(cfg.postgres_dsn)
    else:
        structured = StructuredMemory(cfg.sqlite_path)
    structured.run_startup_migrations()

    semantic = SemanticMemory(cfg.faiss_dim)
    embedding_provider = get_embedding_provider(cfg)
    memory_repository = MemoryRepository(structured, semantic, embedding_provider)

    llm_client = LLMClient(cfg)
    print(f"LLMClient ready — backend: "
          f"{'real (' + cfg.llm_model + ')' if llm_client._real_llm_available else 'mock (deterministic heuristics)'}")

    return {
        "structured": structured,
        "semantic": semantic,
        "embedding_provider": embedding_provider,
        "memory_repository": memory_repository,
        "llm_client": llm_client,
    }


def main() -> None:
    cfg = replace(CFG, use_mock_llm=os.environ.get("GROQ_API_KEY") is None)
    deps = build_dependencies(cfg)

    df, dataset_source = load_dataset(cfg)
    print(f"Running pipeline graph on: {dataset_source}")
    print(f"Dataset shape            : {df.shape}\n")

    graph = build_pipeline_graph(
        cfg=cfg,
        llm_client=deps["llm_client"],
        memory_repository=deps["memory_repository"],
        structured=deps["structured"],
        semantic=deps["semantic"],
    )

    final_state = run_pipeline(df, dataset_source, graph, cfg)

    print()
    print_run_trace(final_state, cfg)

    print("\n" + "-" * 70)
    print("FINAL RESULT")
    print("-" * 70)
    print(f"Problem type      : {final_state['metadata']['problem_type']}")
    print(f"Best model        : {final_state['metrics']['best_model_name']}")
    print(f"Best model metrics: { {k: round(v, 4) for k, v in final_state['metrics']['best_model_metrics'].items()} }")
    print(f"Critic verdict    : {final_state['critic']['recommendation'].upper()}")
    print(f"Critic comments   : {final_state['critic']['comments']}")
    print(f"Experience score  : {final_state['experience_score']['experience_score']:.3f} "
          f"({final_state['experience_score']['recommendation']})")
    print(f"Memory decision   : {final_state['memory_update_decision']['action']} "
          f"— {final_state['memory_update_decision']['reason']}")
    print(f"Total runs stored : {final_state['memory_update_decision'].get('total_runs_stored', 'n/a')}")

    final_state = report_generation_node(final_state, cfg)
    print(f"\nReport written to: {cfg.artifacts_dir}/pipeline_report.md "
          f"({len(final_state['report'])} characters)")

    try:
        get_ipython()  # type: ignore[name-defined]  # noqa: F821
        display_pipeline_outputs(final_state)
    except NameError:
        print("\n(Skipping display_pipeline_outputs — not running inside IPython/Jupyter.)")


if __name__ == "__main__":
    main()