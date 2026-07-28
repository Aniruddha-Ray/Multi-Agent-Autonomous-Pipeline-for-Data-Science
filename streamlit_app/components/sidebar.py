import streamlit as st
import sys
from pathlib import Path 
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from streamlit_app.services.api_client import health
from streamlit_app.services.api_client import health
from streamlit_app.utils.constants import APP_VERSION

def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="sidebar-logo">🧭 Adaptive ML Pipeline</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sidebar-version">{APP_VERSION}</div>', unsafe_allow_html=True)
        st.divider()
        st.subheader("Backend Status")
        try:
            health()
            st.success("Connected")
        except Exception:
            st.error("Disconnected")
        st.divider()
        st.caption(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))