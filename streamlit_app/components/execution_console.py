import streamlit as st

def render_execution_console(logs: list[str] | None = None, error: bool = False):
    st.subheader("Live Execution Console")
    entries = logs or ["Waiting for pipeline execution..."]
    text = "\n".join(entries)
    color = "#ff6b6b" if error else "#d4d4d4"
    st.markdown(f"""
    <div id="exec-console" style="
        background-color:#1e1e1e; color:{color};
        font-family:'Consolas','Courier New',monospace; font-size:13px;
        padding:12px 16px; border-radius:8px; height:340px;
        overflow-y:auto; white-space:pre-wrap; border:1px solid #333;">
{text}
    </div>
    <script>
        var c = document.getElementById("exec-console");
        if (c) {{ c.scrollTop = c.scrollHeight; }}
    </script>
    """, unsafe_allow_html=True)