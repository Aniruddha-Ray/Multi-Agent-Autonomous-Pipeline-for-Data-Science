import streamlit as st
import sys
from pathlib import Path 

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from streamlit_app.components.header import render_header
from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.components.agent_timeline import render_agent_timeline, NODES
from streamlit_app.components.execution_console import render_execution_console
from streamlit_app.components.execution_status import render_execution_status
from streamlit_app.components.metric_cards import render_metric_cards
from streamlit_app.components.report_download import render_report_download
from streamlit_app.components.pipeline_progress import run_simulated_pipeline
from streamlit_app.services.api_client import get_report

render_sidebar()
render_header()
st.header("Adaptive Pipeline Execution")

train_file = st.file_uploader("Upload train.csv", type=["csv"], key="train_file")
test_file = st.file_uploader("Upload test.csv", type=["csv"], key="test_file")
target_column = st.text_input("Target column (leave blank to auto-detect)", value="", key="target_column")
problem_type = st.selectbox("Problem type (leave as Auto-detect to infer automatically)",
                             ["Auto-detect", "Classification", "Regression"])

col_a, col_b = st.columns([1, 1])
run_clicked = col_a.button("Run Pipeline", disabled=not (train_file and test_file))
retry_clicked = col_b.button("Retry") if st.session_state.get("last_error") else False

st.divider()
status_placeholder = st.empty()
timeline_col, console_col = st.columns([1, 2])
timeline_placeholder = timeline_col.empty()
console_placeholder = console_col.empty()

st.subheader("Metrics")
metrics_placeholder = st.empty()
st.subheader("Report")
report_placeholder = st.empty()


def _draw_last_known_state():
    statuses = st.session_state.get("last_statuses") or {key: "pending" for key, _ in NODES}
    logs = st.session_state.get("last_logs") or ["Waiting for pipeline execution..."]
    with timeline_placeholder.container():
        render_agent_timeline(statuses)
    with console_placeholder.container():
        render_execution_console(logs, error=bool(st.session_state.get("last_error")))
    with status_placeholder.container():
        render_execution_status(
            st.session_state.get("last_agent", "Idle"), 0,
            "Error" if st.session_state.get("last_error") else "Idle",
            st.session_state.get("last_run_id"),
        )
    if st.session_state.get("last_metrics"):
        with metrics_placeholder.container():
            render_metric_cards(st.session_state["last_metrics"])
    if st.session_state.get("last_report_text"):
        with report_placeholder.container():
            render_report_download(st.session_state["last_report_text"], st.session_state.get("last_run_id"))


_draw_last_known_state()

if run_clicked or retry_clicked:
    try:
        result = run_simulated_pipeline(
            train_file_bytes=train_file.getvalue(), train_filename=train_file.name,
            test_file_bytes=test_file.getvalue(), test_filename=test_file.name,
            target_column=target_column or None, problem_type=problem_type,
            timeline_placeholder=timeline_placeholder,
            console_placeholder=console_placeholder,
            status_placeholder=status_placeholder,
        )
    except Exception as exc:
        st.session_state["last_error"] = str(exc)
        st.error(f"Pipeline run failed: {exc}")
    else:
        st.session_state["last_error"] = None
        st.session_state["last_run_id"] = result["run_id"]
        st.session_state["last_metrics"] = result["metrics"]
        with metrics_placeholder.container():
            render_metric_cards(result["metrics"])
        with report_placeholder.container():
            try:
                report_text = get_report(result["run_id"])
            except Exception as exc:
                st.error(f"Could not fetch report: {exc}")
            else:
                st.session_state["last_report_text"] = report_text
                render_report_download(report_text, result["run_id"])