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

def get_enhancement(df, shaping_technique):

    df_shaped = df[df["unique_id"] == shaping_technique]
    df_raw = df[df["unique_id"].str.contains("raw")]

    df_shaped = df_shaped.set_index(["dataset_name", "model", "horizon", "metric"])
    df_raw = df_raw.set_index(["dataset_name", "model", "horizon", "metric"])

    df_shaped = df_shaped.drop("unique_id", axis=1)
    df_raw = df_raw.drop("unique_id", axis=1)

    df_shaped = df_shaped.rename({"value": "value_shaped"}, axis=1)
    df_raw = df_raw.rename({"value": "value_raw"}, axis=1)

    comparison_df = pd.concat([df_shaped, df_raw], axis=1)

    comparison_df["shaped_improv"] = comparison_df["value_raw"] - comparison_df["value_shaped"]
    comparison_df = comparison_df.sort_values("shaped_improv", ascending=False)

    comparison_df = comparison_df.drop(["value_shaped", "value_raw"], axis=1)

    return comparison_df.reset_index()