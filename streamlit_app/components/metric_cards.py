import streamlit as st

def render_metric_cards(metrics: dict):
    if not metrics:
        st.info("No metrics yet.")
        return
    cols = st.columns(len(metrics))
    for col, (name, value) in zip(cols, metrics.items()):
        col.metric(name, f"{value:.4f}" if isinstance(value, float) else value)