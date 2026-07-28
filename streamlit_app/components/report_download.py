import streamlit as st

def render_report_download(report_text: str | None, run_id: str | None):
    if not report_text:
        st.info("No report available yet.")
        return
    st.download_button(
        "Download Report (Markdown)",
        data=report_text,
        file_name=f"pipeline_report_{run_id}.md",
        mime="text/markdown",
    )