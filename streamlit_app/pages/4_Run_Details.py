import streamlit as st
import sys
from pathlib import Path 

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from streamlit_app.components.header import render_header
from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.services.api_client import get_run

render_sidebar()
render_header()
st.header("Run Details")

run_id = st.text_input("Run ID")
if st.button("Fetch") and run_id:
    try:
        st.json(get_run(run_id))
    except Exception as exc:
        st.error(f"Could not fetch run: {exc}")