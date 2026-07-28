import streamlit as st

def render_execution_console(logs: list[str] | None = None):
    st.subheader("Adaptive Agent Console")
    default_logs = ["Waiting for pipeline execution..."]
    st.code("\n".join(logs or default_logs), language=None)