"""LangGraph pipeline construction.

Extracted verbatim from Notebook Cell 26 ("CELL 7 — GRAPH CONSTRUCTION")
— node/edge topology is completely unchanged. The only difference from the
notebook is HOW each node function reaches the dependencies it needs:
the notebook closed over module-level globals (llm_client,
memory_repository, structured_memory, semantic_memory); this module binds
the same objects into each node via functools.partial, since Stage E5's
extracted agents take them as explicit parameters (dependency injection)
instead.

NOT included in this module: the notebook's own
``pipeline_graph = build_pipeline_graph()`` instantiation — that call now
requires 5 constructed dependencies that don't exist until Stage E8
(main.py, the composition root). ``build_pipeline_graph`` here is a factory
function only.
"""
from __future__ import annotations

from functools import partial
from typing import Any

from langgraph.graph import END, StateGraph
import sys
from pathlib import Path 

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.agents.critic import critic_agent_node
from src.agents.dataset_analyzer import dataset_analyzer_node
from src.agents.eda import eda_agent_node
from src.agents.experience_scoring import experience_scorer_node
from src.agents.explainability import shap_explainability_node
from src.agents.feature_engineering import feature_engineering_node
from src.agents.memory_retrieval import memory_retrieval_node
from src.agents.model_recommendation import model_recommendation_node
from src.agents.planner import planner_node
from src.agents.training import training_agent_node
from src.config.settings import CFG, Config
from src.memory.repository import MemoryRepository
from src.memory.semantic_store import SemanticMemory
from src.memory.structured_store import StructuredMemory
from src.memory.update_policy import memory_update_policy_node
from src.models.state import PipelineState


def critic_routing_function(state: PipelineState) -> str:
    """Conditional edge after the Critic: loop back to the Planner if a
    revision was requested AND the iteration budget isn't exhausted;
    otherwise proceed to finalize the run via the Experience Scorer and
    Memory Update Policy. This is the hard stop that prevents an infinite
    Planner<->Critic loop.

    Reads ``CFG.max_graph_iterations`` via the module-level global (same
    closure pattern already established and preserved in
    ``core/metadata.py``, ``agents/training.py``, ``agents/explainability.py``
    — not a new parameter here, per the notebook's own original behavior).
    """
    if state.get("needs_revision") and state.get("iteration", 0) < CFG.max_graph_iterations:
        return "revise"
    return "finish"


def build_pipeline_graph(
    cfg: Config,
    llm_client: Any,
    memory_repository: MemoryRepository,
    structured: StructuredMemory,
    semantic: SemanticMemory,
) -> Any:
    """Factory: compile the pipeline graph with every DI'd dependency bound
    into its node via ``functools.partial``.

    ``cfg``/``llm_client``/``memory_repository``/``structured``/``semantic``
    are the exact same 5 objects Stage E8's ``main.py`` will construct once
    (composition root) and pass in here — this function performs no
    construction of its own, only binding + wiring, matching the notebook's
    ``build_pipeline_graph()`` which did the wiring only (the construction
    of ``llm_client``/``memory_repository``/etc. happened in earlier cells).
    """
    graph = StateGraph(PipelineState)

    graph.add_node("dataset_analyzer", dataset_analyzer_node)  # no extra deps (Stage E5.1)
    graph.add_node(
        "memory_retrieval",
        partial(memory_retrieval_node, memory_repository=memory_repository, cfg=cfg),
    )
    graph.add_node(
        "planner",
        partial(planner_node, llm_client=llm_client, cfg=cfg, structured=structured, semantic=semantic),
    )
    graph.add_node("eda_agent", partial(eda_agent_node, llm_client=llm_client, cfg=cfg))
    graph.add_node("feature_engineering", partial(feature_engineering_node, llm_client=llm_client))
    graph.add_node("model_recommendation_agent", partial(model_recommendation_node, llm_client=llm_client))
    graph.add_node("training_agent", training_agent_node)  # no extra deps (CFG global, Stage E5.7)
    graph.add_node("shap_explainability", shap_explainability_node)  # no extra deps (CFG global, Stage E5.8)
    graph.add_node("critic_agent", partial(critic_agent_node, llm_client=llm_client))
    graph.add_node("experience_scorer", experience_scorer_node)  # no extra deps (Stage E5.10)
    graph.add_node(
        "memory_update_policy",
        partial(memory_update_policy_node, memory_repository=memory_repository),
    )
    # NOTE: the original "memory_manager" node (which persisted every run
    # unconditionally) was already removed in the notebook itself, before
    # extraction — its logic is split between experience_scorer_node and
    # memory_update_policy_node, both wired above.

    graph.set_entry_point("dataset_analyzer")
    graph.add_edge("dataset_analyzer", "memory_retrieval")
    graph.add_edge("memory_retrieval", "planner")
    graph.add_edge("planner", "eda_agent")
    graph.add_edge("eda_agent", "feature_engineering")
    graph.add_edge("feature_engineering", "model_recommendation_agent")
    graph.add_edge("model_recommendation_agent", "training_agent")
    graph.add_edge("training_agent", "shap_explainability")
    graph.add_edge("shap_explainability", "critic_agent")
    graph.add_conditional_edges(
        "critic_agent", critic_routing_function,
        {"revise": "planner", "finish": "experience_scorer"},
    )
    graph.add_edge("experience_scorer", "memory_update_policy")
    graph.add_edge("memory_update_policy", END)

    return graph.compile()

def run_pipeline(df, dataset_source, graph, cfg, target_column: str | None = None, problem_type: str | None = None):
    initial_state: PipelineState = {
        "dataset": df,
        "dataset_source": dataset_source,
        "history": [],
    }
    if target_column:
        initial_state["target_column"] = target_column
    if problem_type:
        initial_state["problem_type"] = problem_type
    recursion_budget = 8 * cfg.max_graph_iterations + 10
    final_state = graph.invoke(initial_state, config={"recursion_limit": recursion_budget})
    return final_state