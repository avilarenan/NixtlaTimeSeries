import pandas as pd
from experiments_metadata import *
pd.set_option("display.precision", 8)

def get_results_summary():

    all_metrics_dfs = []

    for dataset in DATASETS:
        for model in MODEL_NAMES_LIST:
            for horizon in HORIZONS:
                for shaping_process in SHAPING_PROCESSES:

                    _shaping_windows = SHAPING_WINDOWS

                    if shaping_process == "identity":
                        _shaping_windows = ["N/A"] # NOTE: must be a single element list only, otherwise identity results will show up multiple times in results summary
                        
                    for shaping_window in _shaping_windows:
                        if shaping_window == "N/A":
                            filename = f"./results/{dataset}_na_{shaping_process}/{model}_horizon{horizon}/metrics.csv"
                        else:
                            filename = f"./results/{dataset}_w{shaping_window}_{shaping_process}/{model}_horizon{horizon}/metrics.csv"
                        df = pd.read_csv(filename)
                        df = df.rename(columns={model: "value"})
                        df["model"] = model
                        df["horizon"] = horizon
                        df["shaping_process"] = shaping_process
                        df["shaping_window"] = shaping_window
                        df["dataset"] = dataset

                        all_metrics_dfs += [df]

    df = pd.concat(all_metrics_dfs)
    df = df[["dataset", "horizon", "shaping_process", "shaping_window", "model", "metric", "value"]]
    df["value"] = pd.to_numeric(df["value"])
    df = df.sort_values(by="value", ascending=False)
    df.to_csv("tmp.csv")
    return df
