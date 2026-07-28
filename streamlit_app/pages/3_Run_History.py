import streamlit as st
import pandas as pd
import sys
from pathlib import Path 

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.components.footer import render_footer
from streamlit_app.services.api_client import get_memory
from streamlit_app.utils.style_loader import load_css

st.set_page_config(page_title="Run History", layout="wide")
load_css()
render_sidebar()
st.markdown('<div class="app-hero-title">Run History</div>', unsafe_allow_html=True)

try:
    runs = get_memory()
    df = pd.DataFrame(runs) if runs else pd.DataFrame()
except Exception as exc:
    st.error(f"Could not reach backend: {exc}")
    df = pd.DataFrame()

if not df.empty:
    search = st.text_input("Search runs", "")
    if search:
        mask = df.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
        df = df[mask]

    sort_col = st.selectbox("Sort by", df.columns.tolist())
    sort_dir = st.radio("Order", ["Descending", "Ascending"], horizontal=True)
    df = df.sort_values(sort_col, ascending=(sort_dir == "Ascending"))

    page_size = 10
    total_pages = max((len(df) - 1) // page_size + 1, 1)
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
    start = (page - 1) * page_size
    st.dataframe(df.iloc[start:start + page_size], use_container_width=True, hide_index=True)
    st.caption(f"Page {page} of {total_pages} — {len(df)} total runs")
else:
    st.info("No runs recorded yet.")

render_footer()