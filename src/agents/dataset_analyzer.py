"""Dataset Analyzer Agent — first LangGraph node in the pipeline.

Extracted verbatim from Notebook Cell 5 ("CELL 6a — LLM CLIENT ... &
DATASET ANALYZER AGENT") — ``dataset_analyzer_node`` only.

Deliberately NOT included in this module:
  - ``LLMClient`` and its instantiation — deferred to Stage F (LLM
    integration). ``dataset_analyzer_node`` itself never calls
    ``llm_client``/``LLMClient`` in the notebook, so this exclusion has no
    behavioral effect on this node.
  - ``_infer_target_column``, ``validate_dataset_for_training``,
    ``compute_dataset_metadata`` — these are defined in this same notebook
    cell but were already extracted in Stage E2 (``core/validation.py``,
    ``core/metadata.py``) from this identical source. Imported from there
    rather than redefined, so exactly one copy exists in the package.
  - The ``_preview_state``/``_meta_preview`` smoke test and its diagnostic
    ``print(...)`` block — notebook-only scratch code, not part of the
    compiled graph.
"""
from __future__ import annotations
import sys
import os
from pathlib import Path 

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.config.settings import CFG
from src.core.metadata import _infer_target_column, compute_dataset_metadata
from src.core.validation import validate_dataset_for_training
from src.models.state import PipelineState


def dataset_analyzer_node(state: PipelineState) -> PipelineState:
    """LangGraph node: validate the dataset, then populate
    ``state['metadata']`` and ``state['target_column']``.

    Validation (Task 7, architecture audit) runs first and raises a clear
    ``PipelineValidationError`` — empty dataset, missing/invalid target
    column, no feature columns, too few samples, or cross-validation not
    being feasible for ``CFG.cv_folds`` — instead of letting the same
    problem surface later as an opaque library error deep inside training.
    """
    df = state["dataset"]
    target_column = state.get("target_column") or _infer_target_column(df)
    validate_dataset_for_training(df, target_column, CFG)

    metadata = compute_dataset_metadata(df, target_column, problem_type=state.get("problem_type"))
    state["target_column"] = metadata["target_column"]
    state["metadata"] = metadata
    state.setdefault("history", []).append({"node": "dataset_analyzer", "status": "ok"})
    return state