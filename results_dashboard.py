import streamlit as st
import pandas as pd
pd.set_option("display.precision", 8)
import plotly.express as px
from results_utils import get_results_summary
from streamlit_helpers import filter_dataframe


st.set_page_config(
    page_title="Results",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded"
)

value_column_config={
    "value": st.column_config.NumberColumn(
        "Value",
        help="Value",
        step=1e-10,
    )
}

st.title("Results")

df = get_results_summary()

st.dataframe(filter_dataframe(df))