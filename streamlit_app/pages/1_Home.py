import streamlit as st
import sys
from pathlib import Path 

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.components.footer import render_footer
from streamlit_app.services.api_client import get_memory
from streamlit_app.utils.style_loader import load_css
from streamlit_app.utils.constants import THEME

st.set_page_config(page_title="Home", layout="wide")
load_css()
render_sidebar()

st.markdown('<div class="app-hero-title">Adaptive Multi-Agent Data Science Pipeline</div>', unsafe_allow_html=True)
st.markdown('<div class="app-hero-subtitle">Autonomous agents that plan, engineer, train, explain, and critique '
            'ML runs — with memory across every run.</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 3])
with col1:
    st.page_link("pages/2_Run_Pipeline.py", label="🚀 Run a Pipeline", use_container_width=True)

st.markdown("".join(
    f'<span class="tech-badge">{b}</span>' for b in
    ["LangGraph", "Groq (Llama)", "PostgreSQL", "FAISS", "FastAPI", "Streamlit"]
), unsafe_allow_html=True)

st.write("")
st.subheader("Platform Overview")

try:
    runs = get_memory()
except Exception:
    runs = []

stat_cols = st.columns(4)
stats = [
    ("Runs Executed", len(runs) if isinstance(runs, list) else "—"),
    ("Models Supported", "10+"),
    ("Memory Backend", "PostgreSQL + FAISS"),
    ("LLM Connected", "Groq (Llama)"),
]
for col, (label, value) in zip(stat_cols, stats):
    with col:
        st.markdown(f"""
        <div class="metric-card" style="--accent:{THEME['primary']}">
            <div class="metric-card-label">{label}</div>
            <div class="metric-card-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

st.write("")
st.subheader("How it works")
feature_cols = st.columns(3)
features = [
    ("📊 Analyze", "Understands your dataset's shape, types, and target automatically."),
    ("🧠 Plan & Train", "Agents plan features, select models, tune, and train autonomously."),
    ("🔁 Learn", "Every run's outcome is stored and informs future runs via semantic memory."),
]
for col, (title, desc) in zip(feature_cols, features):
    with col:
        st.markdown(f'<div class="card"><div class="card-title">{title}</div><div>{desc}</div></div>',
                     unsafe_allow_html=True)

render_footer(backend_status="Connected" if runs is not None else "Unknown")