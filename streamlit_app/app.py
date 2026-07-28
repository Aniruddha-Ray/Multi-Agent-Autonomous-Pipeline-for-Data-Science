import streamlit as st
import sys
from pathlib import Path 

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.utils.style_loader import load_css

st.set_page_config(page_title="Adaptive ML Pipeline", page_icon="🧭", layout="wide")
load_css()
render_sidebar()

st.markdown('<div class="app-hero-title">Adaptive Multi-Agent Data Science Pipeline</div>', unsafe_allow_html=True)
st.markdown('<div class="app-hero-subtitle">Use the sidebar to navigate: Home, Run Pipeline, Run History, Run Details, About.</div>', unsafe_allow_html=True)