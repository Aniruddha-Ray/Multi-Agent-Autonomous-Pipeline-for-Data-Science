"""Central configuration for the Adaptive Multi-Agent Autonomous Data
Science Pipeline.

Extracted verbatim from Notebook Cell 2 ("CELL 2 — CONFIG"). The ``Config``
dataclass's fields and defaults are unchanged from the notebook. The
module-level ``CFG`` singleton and the ``os.makedirs(...)`` call are
preserved as import-time side effects, matching the notebook exactly,
because downstream code (data loading, memory initialization, artifact
writers) depends on ``CFG.artifacts_dir`` already existing.

NOT reproduced here: the notebook's interactive
``print("Config loaded:") ...`` diagnostic loop. That was Jupyter smoke-test
output with no data or control-flow consumers elsewhere in the notebook —
see Stage E1 analysis for the explicit decision not to carry it over.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS_DIR = str(PROJECT_ROOT / "artifacts")


@dataclass(frozen=True)
class Config:
    """Central configuration for the adaptive multi-agent pipeline.
    ...(see docstring for full field descriptions)...
    """
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    n_optuna_trials: int = 20
    max_graph_iterations: int = 3
    high_cardinality_threshold: int = 15
    imbalance_ratio_threshold: float = 1.5
    pca_variance_threshold: float = 0.95
    overfit_gap_threshold: float = 0.12
    use_mock_llm: bool = True
    llm_model: str = "llama-3.3-70b-versatile"
    sqlite_path: str = "pipeline_memory.db"
    faiss_dim: int = 32
    uploads_dir: str = "/mnt/user-data/uploads"
    artifacts_dir: str = "artifacts"
    embedding_provider: str = "local"          # "local" | "openai" | "voyageai" | "gemini"
    memory_retrieval_top_k: int = 5
    memory_min_similarity: float = 0.0         # 0.0 = no threshold filtering


CFG = Config()
ROOT_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = str(ROOT_DIR / CFG.artifacts_dir)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
CFG = Config(artifacts_dir=ARTIFACTS_DIR)
