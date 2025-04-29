import pandas as pd
from pathlib import Path


def get_results():
    results = []

    pathlist = Path("./results/").glob('**/*.csv')
    for path in pathlist:
        horizon = str(path).split("/")[1][7:]
        dataset_name = str(path).split("/")[2].split(".")[0]
        df = pd.read_csv(path)
        df["dataset_name"] = dataset_name
        df["horizon"] = horizon
        results += [df]

    df = pd.concat(results)
    df = df.drop("best_model", axis=1)

    list_dfs_reshaping = []
    basics = ["unique_id", "metric", "dataset_name", "horizon"]
    for column in ["AutoLSTM", "AutoMLP", "AutoNHITS", "AutoTFT", "AutoNBEATSx", "AutoTiDE", "AutoBiTCN", "AutoDeepNPTS"]:
        df_tmp = df[basics + [column]]
        df_tmp = df_tmp.rename({column: "value"}, axis=1)
        df_tmp["model"] = column
        list_dfs_reshaping += [df_tmp]

    return pd.concat(list_dfs_reshaping)