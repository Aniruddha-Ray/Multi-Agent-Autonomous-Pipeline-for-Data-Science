import streamlit as st
import sys
from pathlib import Path 

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from streamlit_app.components.sidebar import render_sidebar

st.set_page_config(page_title="Adaptive ML Pipeline", layout="wide")
render_sidebar()

st.title("Adaptive Multi-Agent Data Science Pipeline")
st.write("Use the sidebar navigation to explore: Home, Run Pipeline, Run History, Run Details, About.")