from datasetsforecast.long_horizon2 import LongHorizon2
# from datasetsforecast.long_horizon import LongHorizon
import pandas as pd
pd.options.plotting.backend = "plotly"
import warnings
warnings.filterwarnings('ignore')
from PyFARM import farm
from tqdm.notebook import tqdm

from datasets_metadata import ts_metadata

PLOT = False
datasets_names = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "ECL", "TrafficL"]

for dataset_name in tqdm(datasets_names):
    Y_df = LongHorizon2.load(directory='data', group=dataset_name)

    target_ts = ts_metadata[dataset_name]["target_ts"]
    exog_list = ts_metadata[dataset_name]["exog_list"]
    test_size = ts_metadata[dataset_name]["test_size"]
    valid_size = ts_metadata[dataset_name]["valid_size"]
    farm_windows = ts_metadata[dataset_name]["farm_windows"]

    Y_df["ds"] = pd.to_datetime(Y_df["ds"])
    Y_df = Y_df.set_index("ds")

    # Pivotting original dataset in order to consider other series as exogenous
    df_raw = Y_df[Y_df["unique_id"] == target_ts]
    for item in tqdm(Y_df["unique_id"].unique()):
        if item == target_ts:
            continue
        df_raw[item] = Y_df[Y_df["unique_id"] == item]["y"]
    df_raw = df_raw.reset_index()
    df_raw = df_raw.drop("index", axis=1)
    df_raw["unique_id"] = f"{target_ts}_raw"

    # FARM SHAPING
    ref_ts = "y"

    list_of_dfs = []
    for farm_window in tqdm(farm_windows):
        df_shaped = df_raw.copy()
        for feature in tqdm(exog_list):
            qts_shaped = farm(
                refTS=df_raw[ref_ts].values,
                qryTS=df_raw[str(feature)].values,
                ff_align=False,
                lcwin=farm_window,
                fuzzyc=[1]
            )["qts_shaped"]
            df_shaped[feature] = qts_shaped

        df_shaped["unique_id"] = f"{target_ts}_w{farm_window}_exogenous_FARM_shaped"
        list_of_dfs += [df_shaped]
    df_shaped = pd.concat(list_of_dfs)

    # ROLLING COVARIANCE
    ref_ts = "y"

    list_of_dfs = []
    for farm_window in tqdm(farm_windows):
        df_cov = df_raw.copy()
        for feature in tqdm(exog_list):
            qts_shaped = df_raw[ref_ts].rolling(farm_window).cov(df_raw[str(feature)])
            df_cov[feature] = qts_shaped

        df_cov["unique_id"] = f"{target_ts}_w{farm_window}_exogenous_FARM_shaped"
        list_of_dfs += [df_cov]
    df_cov = pd.concat(list_of_dfs)

    # ROLLING CORRELATION
    ref_ts = "y"

    list_of_dfs = []
    for farm_window in tqdm(farm_windows):
        df_cov = df_raw.copy()
        for feature in tqdm(exog_list):
            qts_shaped = df_raw[ref_ts].rolling(farm_window).corr(df_raw[str(feature)])
            df_cov[feature] = qts_shaped

        df_cov["unique_id"] = f"{target_ts}_w{farm_window}_exogenous_FARM_shaped"
        list_of_dfs += [df_cov]
    df_cov = pd.concat(list_of_dfs)

    # TODO: relative entropy, distances

    df = pd.concat([df_raw, df_shaped, df_cov])
    # df.to_csv(f"./processed_data/{dataset_name}.csv", index=False)
    df.to_parquet(f"./processed_data/{dataset_name}.parquet", index=False)