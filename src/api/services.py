import time
import uuid
import sys
import os
from pathlib import Path 

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.main import run_end_to_end

# In-memory run store — process-local, does not survive a restart.
# See open note below on making this durable.
_RUN_STORE: dict[str, dict] = {}


def execute_pipeline_run(df, dataset_source: str, cfg, deps) -> dict:
    run_id = str(uuid.uuid4())
    start = time.perf_counter()

    final_state = run_end_to_end(df, dataset_source, cfg, deps)  # let it raise; route layer converts to HTTP

    result = {
        "run_id": run_id,
        "status": "completed",
        "best_model": final_state["metrics"]["best_model_name"],
        "metrics": final_state["metrics"]["best_model_metrics"],
        "report_path": f"{cfg.artifacts_dir}/pipeline_report.md",  # informational only — see note
        "execution_time": time.perf_counter() - start,
        "report_text": final_state["report"],  # captured per-run, since the file itself gets overwritten
    }
    _RUN_STORE[run_id] = result
    return result


def get_memory_snapshot(deps, limit: int | None = None) -> list[dict]:
    """Thin passthrough to the existing MemoryRepository.list_memories() —
    no new read path, just exposes what already exists over HTTP."""
    return deps["memory_repository"].list_memories(limit=limit)