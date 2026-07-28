import streamlit as st

def render_execution_status(current_agent: str, elapsed_seconds: float, backend_status: str, run_id: str | None = None):
    cols = st.columns(4)
    cols[0].metric("Current Agent", current_agent)
    mins, secs = divmod(int(elapsed_seconds), 60)
    cols[1].metric("Elapsed", f"{mins:02d}:{secs:02d}")
    cols[2].metric("Backend", backend_status)
    cols[3].metric("Run ID", run_id or "—")