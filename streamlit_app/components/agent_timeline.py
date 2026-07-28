import streamlit as st

NODES = [
    ("dataset_analyzer", "Dataset Analyzer"),
    ("memory_retrieval", "Memory Retrieval"),
    ("planner", "Planner"),
    ("eda_agent", "EDA"),
    ("feature_engineering", "Feature Engineering"),
    ("model_recommendation_agent", "Model Recommendation"),
    ("training_agent", "Training"),
    ("shap_explainability", "Explainability"),
    ("critic_agent", "Critic"),
    ("experience_scorer", "Experience Scorer"),
    ("memory_update_policy", "Memory Update"),
]

_STATUS_STYLE = {
    "pending":   ("Pending",   "#9e9e9e"),
    "running":   ("Running",   "#1f77b4"),
    "completed": ("Completed", "#2ca02c"),
    "failed":    ("Failed",    "#d62728"),
}

def render_agent_timeline(statuses: dict[str, str]):
    st.subheader("Agent Timeline")
    rows = []
    for key, label in NODES:
        state = statuses.get(key, "pending")
        status_word, color = _STATUS_STYLE.get(state, _STATUS_STYLE["pending"])
        weight = "700" if state == "running" else "400"
        rows.append(
            f'<div style="display:flex;align-items:center;gap:8px;padding:4px 0;'
            f'font-family:inherit;color:{color} !important;font-weight:{weight};">'
            f'<span style="width:10px;height:10px;border-radius:50%;'
            f'background-color:{color};display:inline-block;flex-shrink:0;"></span>'
            f'<span>{label}</span>'
            f'<span style="font-size:11px;opacity:0.75;">({status_word})</span>'
            f'</div>'
        )
    st.markdown("".join(rows), unsafe_allow_html=True)