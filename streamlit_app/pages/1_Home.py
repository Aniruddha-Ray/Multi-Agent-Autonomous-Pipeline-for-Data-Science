import streamlit as st
import sys
from pathlib import Path 

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from streamlit_app.components.header import render_header
from streamlit_app.components.footer import render_footer
from streamlit_app.components.sidebar import render_sidebar

render_sidebar()
render_header()

st.subheader("About this project")
st.write("An autonomous multi-agent pipeline that plans, engineers features, trains, "
         "explains, and critiques its own ML runs — with persistent memory across runs.")

st.subheader("Technology stack")
st.write("- LangGraph (multi-agent orchestration)\n"
         "- Groq (Llama-based LLM backend)\n"
         "- PostgreSQL (structured run memory)\n"
         "- FAISS (semantic memory retrieval)\n"
         "- FastAPI (backend REST API)\n"
         "- Streamlit (this frontend)")

st.subheader("Architecture")
st.info("Architecture diagram placeholder — added in a later stage.")

render_footer()