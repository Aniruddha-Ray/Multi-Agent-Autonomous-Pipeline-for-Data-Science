import streamlit as st
import sys
from pathlib import Path 

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from streamlit_app.services.api_client import health

def render_sidebar():
    with st.sidebar:
        st.subheader("Backend Status")
        try:
            health()
            st.success("Connected")
        except Exception:
            st.error("Disconnected")