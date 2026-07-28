import streamlit as st

def render_prediction_download(prediction_bytes: bytes | None, run_id: str | None):
    if not prediction_bytes:
        st.info("No predictions available yet.")
        return
    st.download_button(
        "Download Predictions (CSV)",
        data=prediction_bytes,
        file_name=f"prediction_{run_id}.csv",
        mime="text/csv",
    )