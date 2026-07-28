import streamlit as st
import sys
from pathlib import Path 

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from streamlit_app.components.header import render_header
from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.components.execution_console import render_execution_console
from streamlit_app.components.metric_cards import render_metric_cards
from streamlit_app.components.report_download import render_report_download
from streamlit_app.services.api_client import run_pipeline, get_report

render_sidebar()
render_header()
st.header("Run Pipeline")

train_file = st.file_uploader("Upload train.csv", type=["csv"], key="train_file")
test_file = st.file_uploader("Upload test.csv", type=["csv"], key="test_file")
target_column = st.text_input(
    "Target column (leave blank to auto-detect)", value="", key="target_column"
)
problem_type = st.selectbox(
    "Problem type (leave as Auto-detect to infer automatically)",
    ["Auto-detect", "Classification", "Regression"],
)

run_clicked = st.button("Run Pipeline", disabled=not (train_file and test_file))

console_placeholder = st.empty()
metrics_placeholder = st.empty()
report_placeholder = st.empty()

with console_placeholder.container():
    render_execution_console()
with metrics_placeholder.container():
    st.subheader("Metrics")
    render_metric_cards({})
with report_placeholder.container():
    st.subheader("Report")
    render_report_download(None, None)

if run_clicked:
    with console_placeholder.container():
        render_execution_console(["Uploading files...", "Running pipeline — this can take a while..."])

    try:
        result = run_pipeline(
            train_file_bytes=train_file.getvalue(),
            train_filename=train_file.name,
            test_file_bytes=test_file.getvalue(),
            test_filename=test_file.name,
            target_column=target_column or None,
            problem_type=problem_type,
        )
    except Exception as exc:
        with console_placeholder.container():
            render_execution_console([f"Pipeline run failed: {exc}"])
    else:
        with console_placeholder.container():
            render_execution_console([
                f"Run ID: {result['run_id']}",
                f"Status: {result['status']}",
                f"Target column used: {result['target_column']}",
                f"Problem type used: {result['problem_type']}",
                f"Best model: {result['best_model']}",
                f"Execution time: {result['execution_time']:.2f}s",
                f"Predictions written to: {result.get('prediction_path', 'n/a')}",
            ])
        with metrics_placeholder.container():
            st.subheader("Metrics")
            render_metric_cards(result["metrics"])
        with report_placeholder.container():
            st.subheader("Report")
            try:
                report_text = get_report(result["run_id"])
            except Exception as exc:
                st.error(f"Could not fetch report: {exc}")
            else:
                render_report_download(report_text, result["run_id"])