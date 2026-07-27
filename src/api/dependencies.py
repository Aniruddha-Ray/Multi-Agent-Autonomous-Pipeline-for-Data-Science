from functools import lru_cache
import sys
import os
from pathlib import Path 

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.main import build_dependencies, get_runtime_config

@lru_cache(maxsize=1)
def get_deps():
    """Build once, reuse for the life of the process — the LLM client,
    Postgres connection, and FAISS index are expensive to construct and
    must not be rebuilt per-request."""
    cfg = get_runtime_config()
    return cfg, build_dependencies(cfg)