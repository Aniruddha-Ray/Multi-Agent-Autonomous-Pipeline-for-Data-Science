import threading
import time
from datetime import datetime

from components.agent_timeline import render_agent_timeline, NODES
from components.execution_console import render_execution_console
from components.execution_status import render_execution_status
from services.api_client import run_pipeline

def _ts(msg: str) -> str:
    return f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"

# (node_key, log lines shown during this stage, simulated duration in seconds)
STAGE_SCRIPT = [
    ("dataset_analyzer", ["Reading CSV...", "Running Dataset Analyzer...", "Target column detected"], 3),
    ("memory_retrieval", ["Retrieving similar memories...", "Semantic search complete"], 2),
    ("planner", ["Planner selecting candidate models..."], 3),
    ("eda_agent", ["Running exploratory data analysis..."], 2),
    ("feature_engineering", ["Engineering features..."], 2),
    ("model_recommendation_agent", ["Ranking candidate models..."], 2),
    ("training_agent", ["Running hyperparameter search (Optuna)...", "Training best model..."], 8),
    ("shap_explainability", ["Generating SHAP explanations..."], 3),
    ("critic_agent", ["Critic reviewing run quality..."], 2),
    ("experience_scorer", ["Scoring run experience..."], 1),
    ("memory_update_policy", ["Updating memory store..."], 1),
]


def run_simulated_pipeline(
    train_file_bytes, train_filename, test_file_bytes, test_filename,
    target_column, problem_type,
    timeline_placeholder, console_placeholder, status_placeholder,
) -> dict:
    """Runs the real API call on a background thread while simulating
    realistic per-stage progress. Once the real response lands, remaining
    stages are resolved (completed/failed) from the actual outcome, and the
    real result dict is returned (or the real exception re-raised)."""

    result_box: dict = {}

    def _worker():
        try:
            result_box["result"] = run_pipeline(
                train_file_bytes=train_file_bytes, train_filename=train_filename,
                test_file_bytes=test_file_bytes, test_filename=test_filename,
                target_column=target_column, problem_type=problem_type,
            )
        except Exception as exc:
            result_box["error"] = exc

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    statuses = {key: "pending" for key, _ in NODES}
    logs = [_ts("Dataset received"), _ts("Uploading files to backend...")]
    start = time.time()

    def _draw(current_label: str, backend_state: str = "Running"):
        with timeline_placeholder.container():
            render_agent_timeline(statuses)
        with console_placeholder.container():
            render_execution_console(logs)
        with status_placeholder.container():
            render_execution_status(current_label, time.time() - start, backend_state)

    _draw("Starting")

    for node_key, lines, duration in STAGE_SCRIPT:
        if "result" in result_box or "error" in result_box:
            break
        statuses[node_key] = "running"
        label = dict(NODES)[node_key]
        logs.append(_ts(f"Running {label}..."))
        _draw(label)

        step_elapsed, line_idx, tick = 0.0, 0, 0.5
        while step_elapsed < duration:
            if "result" in result_box or "error" in result_box:
                break
            time.sleep(tick)
            step_elapsed += tick
            if line_idx < len(lines) and step_elapsed >= (line_idx + 1) * (duration / max(len(lines), 1)):
                logs.append(_ts(lines[line_idx]))
                line_idx += 1
                _draw(label)

        if "result" not in result_box and "error" not in result_box:
            statuses[node_key] = "completed"
            _draw(label)

    LOOP_SEQUENCE = ["planner", "model_recommendation_agent", "training_agent", "shap_explainability", "critic_agent", "experience_scorer", "memory_update_policy"]
    LOOP_MESSAGES = {
        "planner": "Planner reconsidering model/feature strategy...",
        "model_recommendation_agent": "Re-ranking candidate models...",
        "training_agent": "Re-training candidate pipeline...",
        "shap_explainability": "Re-generating SHAP explanations...",
        "critic_agent": "Critic re-evaluating revised run...",
        "experience_scorer": "Scoring experience...",
        "memory_update_policy": "Checking memory for similar past revisions...",
    }

    loop_step = 0
    while thread.is_alive():
        active_node = LOOP_SEQUENCE[loop_step % len(LOOP_SEQUENCE)]
        prev_node = LOOP_SEQUENCE[(loop_step - 1) % len(LOOP_SEQUENCE)]

        for node in LOOP_SEQUENCE:
            statuses[node] = "completed" if node == prev_node else (
                "running" if node == active_node else statuses.get(node, "pending")
            )

        logs.append(_ts(LOOP_MESSAGES[active_node]))
        loop_step += 1
        _draw(f"{dict(NODES)[active_node]} (revision loop)")
        time.sleep(3.5)

    thread.join()

    if "error" in result_box:
        for key in statuses:
            if statuses[key] == "running":
                statuses[key] = "failed"
        logs.append(_ts(f"ERROR: {result_box['error']}"))
        with timeline_placeholder.container():
            render_agent_timeline(statuses)
        with console_placeholder.container():
            render_execution_console(logs, error=True)
        with status_placeholder.container():
            render_execution_status("Failed", time.time() - start, "Error")
        raise result_box["error"]

    for key in statuses:
        statuses[key] = "completed"
    result = result_box["result"]
    logs.append(_ts(f"Best model: {result['best_model']}"))
    logs.append(_ts(f"Pipeline completed successfully. Run ID: {result['run_id']}"))
    with timeline_placeholder.container():
        render_agent_timeline(statuses)
    with console_placeholder.container():
        render_execution_console(logs)
    with status_placeholder.container():
        render_execution_status("Completed", time.time() - start, "Idle", result["run_id"])

    return result