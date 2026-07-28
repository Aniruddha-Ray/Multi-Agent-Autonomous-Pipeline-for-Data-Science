import streamlit as st
import sys
from pathlib import Path 

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from streamlit_app.components.header import render_header
from streamlit_app.components.footer import render_footer
from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.utils.style_loader import load_css

load_css()
render_sidebar()
render_header()
st.header("About")
st.write("Built to automate exploratory data analysis, feature engineering, model "
         "selection, training, explainability, and critique — with memory of past runs "
         "informing future ones.")
render_footer()