import streamlit as st
import pandas as pd
import sys
from pathlib import Path 

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from streamlit_app.components.header import render_header
from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.services.api_client import get_memory

render_sidebar()
render_header()
st.header("Run History")

try:
    runs = get_memory()
    st.dataframe(pd.DataFrame(runs) if runs else pd.DataFrame())
except Exception as exc:
    st.error(f"Could not reach backend: {exc}")