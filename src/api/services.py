import time
import uuid
import sys
from pathlib import Path 

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.main import run_end_to_end

# In-memory run store — process-local, does not survive a restart.
# See open note below on making this durable.
_RUN_STORE: dict[str, dict] = {}


def execute_pipeline_run(
    train_df, test_df, dataset_source: str, target_column: str | None, cfg, deps,
    problem_type: str | None = None,
) -> dict:
    run_id = str(uuid.uuid4())
    start = time.perf_counter()

    final_state = run_end_to_end(
        train_df, dataset_source, cfg, deps,
        target_column=target_column, problem_type=problem_type,
    )

    best_name = final_state["metrics"]["best_model_name"]
    best_pipeline = final_state["models"]["trained_pipelines"][best_name]
    test_features = test_df.drop(columns=[target_column], errors="ignore")
    preds = best_pipeline.predict(test_features)

    result_df = test_df.copy()
    result_df[target_column] = preds
    pred_path = f"{cfg.artifacts_dir}/prediction_{run_id}.csv"
    result_df.to_csv(pred_path, index=False)

    result = {
        "run_id": run_id,
        "status": "completed",
        "best_model": best_name,
        "metrics": final_state["metrics"]["best_model_metrics"],
        "report_path": f"{cfg.artifacts_dir}/pipeline_report.md",
        "execution_time": time.perf_counter() - start,
        "report_text": final_state["report"],
        "prediction_path": pred_path,
        "target_column": final_state["target_column"],
        "problem_type": final_state["metadata"]["problem_type"],
        "details": {
            "dataset": {"source": dataset_source, "n_rows": final_state["metadata"]["n_rows"],
                        "n_cols": final_state["metadata"]["n_cols"]},
            "planner": final_state.get("planner_decision", {}),
            "feature_engineering": {k: v for k, v in final_state.get("features", {}).items() if k != "preprocessor"},
            "training": {"best_model": best_name, "metrics": final_state["metrics"]["best_model_metrics"]},
            "critic": final_state.get("critic", {}),
            "memory": final_state.get("memory_update_decision", {}),
        },
    }
    _RUN_STORE[run_id] = result
    return result


def get_memory_snapshot(deps, limit: int | None = None) -> list[dict]:
    """Thin passthrough to the existing MemoryRepository.list_memories() —
    no new read path, just exposes what already exists over HTTP."""
    return deps["memory_repository"].list_memories(limit=limit)