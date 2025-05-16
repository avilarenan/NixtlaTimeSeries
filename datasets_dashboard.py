import plotly.graph_objects as go; import numpy as np
from plotly_resampler import FigureResampler
from dash import Dash, Input, Output, callback_context, dcc, html, no_update
import dash_bootstrap_components as dbc
import pandas as pd
from pathlib import Path
from dash import Dash, dcc, html, Input, Output,callback

def get_datasets():
    datasets_path = []

    pathlist = Path("./processed_data/").glob('**/*.parquet')
    for path in pathlist:
        # print(path)
        datasets_path += [str(path)]

    return datasets_path

available_datasets = get_datasets()
available_exogenous = []
available_unique_ids = []

df = None

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

fig = FigureResampler()
app.layout = html.Div([
    html.H1("Datasets inspector"),
    html.Br(),
    dcc.Dropdown(available_datasets, value=available_datasets[0], id='datasets-dropdown'),
    html.Br(),
    dbc.Row(
        [
            dbc.Col([
                dbc.Row([
                    dcc.Dropdown(available_unique_ids, id="unique-id-dropdown")
                ]),
                dbc.Row([
                    dcc.Dropdown(available_exogenous, id="exogenous-dropdown"),
                ])
            ]),
            dbc.Col([
                dbc.Row([
                    dcc.Dropdown(available_unique_ids, id="unique-id-dropdown2"),
                ]),
                dbc.Row([
                    dcc.Dropdown(available_exogenous, id="exogenous-dropdown2"),
                ])
            ]),
            dbc.Col([
                dbc.Row([
                    dcc.Dropdown(available_unique_ids, id="unique-id-dropdown3"),
                ]),
                dbc.Row([
                    dcc.Dropdown(available_exogenous, id="exogenous-dropdown3"),
                ])
            ])
        ]
    ),
    # dcc.Dropdown(available_unique_ids, id="unique-id-dropdown"),
    # dcc.Dropdown(available_exogenous, id="exogenous-dropdown"),
    # dcc.Dropdown(available_unique_ids, id="unique-id-dropdown2"),
    # dcc.Dropdown(available_exogenous, id="exogenous-dropdown2"),
    # dcc.Dropdown(available_unique_ids, id="unique-id-dropdown3"),
    # dcc.Dropdown(available_exogenous, id="exogenous-dropdown3"),
    dcc.Graph(id="main-graph", style={'width': '100vw', 'height': '800px'})

])

df = pd.read_parquet("./processed_data/ETTh2.parquet")

@callback(
    Output("exogenous-dropdown", 'options'),
    Output("unique-id-dropdown", 'options'),
    Output("exogenous-dropdown2", 'options'),
    Output("unique-id-dropdown2", 'options'),
    Output("exogenous-dropdown3", 'options'),
    Output("unique-id-dropdown3", 'options'),
    Input('datasets-dropdown', 'value'),
)
def update_chosen_datasets(value):
    global df
    print(f"./{value}")
    df = pd.read_parquet(f"./{value}")

    return df.columns, df["unique_id"].unique(), df.columns, df["unique_id"].unique(), df.columns, df["unique_id"].unique()

@callback(
    Output('main-graph', 'figure'),
    Input('exogenous-dropdown', 'value'),
    Input("unique-id-dropdown", 'value'),
    Input("exogenous-dropdown2", 'value'),
    Input("unique-id-dropdown2", 'value'),
    Input("exogenous-dropdown3", 'value'),
    Input("unique-id-dropdown3", 'value'),
)
def update_output(exogenous, unique_id, exogenous2, unique_id2, exogenous3, unique_id3):

    ctx = callback_context
    if len(ctx.triggered):
        global fig, df
        if len(fig.data):
            fig.replace(go.Figure())

        df1 = df[df["unique_id"] == unique_id]
        df2 = df[df["unique_id"] == unique_id2]
        df3 = df[df["unique_id"] == unique_id3]
        fig.add_trace(go.Scattergl(name=f"{exogenous}_1", showlegend=True), hf_x=df1["ds"], hf_y=df1[exogenous])
        fig.add_trace(go.Scattergl(name=f"{exogenous2}_2", showlegend=True), hf_x=df2["ds"], hf_y=df2[exogenous2])
        fig.add_trace(go.Scattergl(name=f"{exogenous3}_3", showlegend=True), hf_x=df3["ds"], hf_y=df3[exogenous3])
        
        return fig
    else:
        return no_update

fig.register_update_graph_callback(app=app, graph_id="main-graph")


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
