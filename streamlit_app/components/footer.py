import streamlit as st
from datetime import datetime
from streamlit_app.utils.constants import APP_VERSION, AUTHOR_NAME, AUTHOR_GITHUB

def render_footer(backend_status: str = "Unknown", db: str = "PostgreSQL", llm: str = "Groq (Llama)"):
    st.markdown(f"""
    <div class="app-footer">
        <span>Version {APP_VERSION}</span>
        <span>Backend: {backend_status}</span>
        <span>DB: {db}</span>
        <span>LLM: {llm}</span>
        <span>{datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
        <span>Author: {AUTHOR_NAME} · <a href="{AUTHOR_GITHUB}" target="_blank">GitHub</a></span>
    </div>
    """, unsafe_allow_html=True)