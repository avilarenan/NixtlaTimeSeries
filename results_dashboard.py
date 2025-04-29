import streamlit as st
import pandas as pd
import plotly.express as px
from results_utils import get_results

st.set_page_config(
    page_title="Results",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Results")

df = get_results()

st.write("Raw")
st.write(df)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    unique_id_filter = st.multiselect("unique_id_filter", options=df["unique_id"].unique())
df = df[df["unique_id"].isin(unique_id_filter)]

with col2:
    metric_filter = st.multiselect("metric_filter", options=df["metric"].unique())
df = df[df["metric"].isin(metric_filter)]

with col3:
    dataset_filter = st.multiselect("dataset_name_filter", options=df["dataset_name"].unique())
df = df[df["dataset_name"].isin(dataset_filter)]

with col4:
    horizon_filter = st.multiselect("horizon_filter", options=df["horizon"].unique())
df = df[df["horizon"].isin(horizon_filter)]

with col5:
    model_filter = st.multiselect("model_filter", options=df["model"].unique())
df = df[df["model"].isin(model_filter)]

st.write("Filtered")
st.write(df)



graph_style = st.selectbox("Graph", options=["Scatter", "Bar"])

if graph_style == "Scatter":
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        x_axis = st.selectbox("X axis", options=df.columns)
    with col2:
        y_axis = st.selectbox("Y axis", options=df.columns)
    with col3:
        color = st.selectbox("Color", options=df.columns)
    with col4:
        size = st.selectbox("Size", options=df.columns)
    with col5:
        hover = st.multiselect("Hover data", options=df.columns)
    fig = px.scatter(
        df,
        x=x_axis,
        y=y_axis,
        color=color,
        size=size,
        hover_data=hover,
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        x_axis = st.selectbox("X axis", options=df.columns)
    with col2:
        y_axis = st.selectbox("Y axis", options=df.columns)
    with col3:
        color = st.selectbox("Color", options=df.columns)
    with col4:
        hover = st.multiselect("Hover data", options=df.columns)

    fig = px.bar(
        df,
        x=x_axis,
        y=y_axis,
        color=color,
        hover_data=hover,
        barmode="group"
    )
    st.plotly_chart(fig, use_container_width=True)