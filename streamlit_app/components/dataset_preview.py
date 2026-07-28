import streamlit as st
import pandas as pd

def render_dataset_preview(df: pd.DataFrame, label: str):
    st.write(f"**{label}** — {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head(10))