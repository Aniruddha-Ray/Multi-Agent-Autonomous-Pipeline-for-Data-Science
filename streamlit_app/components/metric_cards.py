import streamlit as st
from streamlit_app.utils.constants import THEME

_ICONS = {"accuracy": "🎯", "precision": "🔍", "recall": "📡", "f1": "⚖️", "roc_auc": "📈",
          "rmse": "📉", "r2": "📊", "mae": "📐"}

def render_metric_cards(metrics: dict):
    if not metrics:
        st.info("No metrics yet.")
        return
    cols = st.columns(len(metrics))
    for col, (name, value) in zip(cols, metrics.items()):
        icon = _ICONS.get(name.lower(), "📌")
        display_val = f"{value:.4f}" if isinstance(value, float) else value
        with col:
            st.markdown(f"""
            <div class="metric-card" style="--accent:{THEME['primary']}">
                <div class="metric-card-label">{icon} {name}</div>
                <div class="metric-card-value">{display_val}</div>
            </div>
            """, unsafe_allow_html=True)