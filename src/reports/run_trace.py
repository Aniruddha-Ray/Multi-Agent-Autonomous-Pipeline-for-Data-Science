"""Execution trace printer.

Extracted verbatim from Notebook Cell 27 ("CELL 8 — EXECUTE GRAPH")
— ``print_run_trace`` only.
"""
from __future__ import annotations
import sys
import os
from pathlib import Path 

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.config.settings import Config
from src.models.state import PipelineState


def print_run_trace(state: PipelineState, cfg: Config) -> None:
    """Print a human-readable trace of every node the graph visited,
    including revision-loop iterations, ending with the final verdict.

    ``cfg`` is an explicit parameter here (the notebook read the module
    global ``CFG`` directly) since this is a standalone reporting utility
    with no other dependencies to justify a global-closure exception.
    """
    print("Execution trace:")
    for step, event in enumerate(state.get("history", []), start=1):
        extra = {k: v for k, v in event.items() if k not in ("node", "status")}
        extra_str = f" ({extra})" if extra else ""
        print(f"  {step:2d}. {event['node']:28s} status={event['status']}{extra_str}")

    print(f"\nTotal Planner<->Critic iterations : {state.get('iteration', 0)} "
          f"(cap = {cfg.max_graph_iterations})")
    print(f"Final needs_revision               : {state.get('needs_revision')}")
    stopped_reason = (
        "Critic approved the pipeline."
        if not state.get("needs_revision")
        else f"Iteration budget (cfg.max_graph_iterations={cfg.max_graph_iterations}) exhausted "
             f"while Critic still requested revision."
    )
    print(f"Why the loop stopped               : {stopped_reason}")