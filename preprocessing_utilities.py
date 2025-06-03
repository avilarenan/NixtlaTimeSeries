import warnings
from pathlib import Path

from PyFARM import farm

from dtaidistance import dtw

from pyinform.relativeentropy import relative_entropy
from pyinform.mutualinfo import mutual_info
from pyinform.utils.coalesce import coalesce_series

import numpy as np
import pandas as pd

from numpy.lib.stride_tricks import sliding_window_view

warnings.filterwarnings('once')

def read_df_from_file(path, filename, format):
    if format == ".parquet":
        print(f"Reading from {path}/{filename}.parquet")
        return pd.read_parquet(f"{path}/{filename}.parquet")
    elif format == ".csv":
        print(f"Reading from {path}/{filename}.csv")
        return pd.read_csv(f"{path}/{filename}.csv")
    else:
        raise Exception(f"Unrecognized input file format: {format}. \nAvailable formats are: .csv and .parquet")

def save_df_to_file(df, path, filename, format=".parquet"):
    if any(substring in filename for substring in [".csv", ".parquet"]):
        raise Exception("Filename must not contain .csv or .parquet extensions. Rather use the format parameter.")
    Path(path).mkdir(parents=True, exist_ok=True)
    if format == ".parquet":
        print(f"Saving to {path}/{filename}.parquet")
        df.to_parquet(f"{path}/{filename}.parquet")
    elif format == ".csv":
        print(f"Saving to {path}/{filename}.csv")
        df.to_csv(f"{path}/{filename}.csv", index=False)
    else:
        raise Exception(f"Unrecognized output file format: {format}. \nAvailable formats are: .csv and .parquet")

def pfarm(farm_params):
    '''
    FARM SHAPING
    farm_params = {
        "df_raw" : df_raw,
        "window" : window,
        "exogenous_feature": feature,
        "target_feature": target
    }
    '''
    df_raw = farm_params["df_raw"]
    window = farm_params["window"]
    exogenous_feature = str(farm_params["exogenous_feature"])
    target_feature = str(farm_params["target_feature"])
    ret = farm(
        refTS=df_raw[target_feature].values,
        qryTS=df_raw[str(exogenous_feature)].values,
        ff_align=False,
        lcwin=window,
        fuzzyc=[1]
    )["qts_shaped"]

    return {"shaped" : ret}, exogenous_feature

def prollcorr(params):
    '''
    CORRELATION SHAPING
    params = {
        "df_raw" : df_raw,
        "window" : window,
        "exogenous_feature": feature,
        "target_feature": target
    }
    '''
    df_raw = params["df_raw"]
    window = params["window"]
    exogenous_feature = str(params["exogenous_feature"])
    target_feature = str(params["target_feature"])
    saliency = df_raw[target_feature].rolling(window).corr(df_raw[str(exogenous_feature)])
    
    shaping_ratio = (saliency-saliency.min())/(saliency.max() - saliency.min()) # normalizing between 0 and 1
    shaping_ratio_inverted = (shaping_ratio - 1).abs() # NOTE: INVERTING

    shaping_ratio = shaping_ratio.fillna(1) # NOTE: keep as it is if we can't calculate a ratio (NaN case)
    shaping_ratio_inverted = shaping_ratio_inverted.fillna(1) # NOTE: keep as it is if we can't calculate a ratio (NaN case)

    ret = df_raw[str(exogenous_feature)] * shaping_ratio
    ret_inverted = df_raw[str(exogenous_feature)] * shaping_ratio_inverted

    return {"shaped" : ret, "inverted_shaped": ret_inverted}, exogenous_feature

def prollcov(params):
    '''
    COVARIANCE SHAPING
    params = {
        "df_raw" : df_raw,
        "window" : window,
        "exogenous_feature": feature,
        "target_feature": target
    }
    '''
    df_raw = params["df_raw"]
    window = params["window"]
    exogenous_feature = str(params["exogenous_feature"])
    target_feature = str(params["target_feature"])
    saliency = df_raw[target_feature].rolling(window).cov(df_raw[str(exogenous_feature)])

    shaping_ratio = (saliency-saliency.min())/(saliency.max() - saliency.min()) # normalizing between 0 and 1
    shaping_ratio_inverted = (shaping_ratio - 1).abs() # NOTE: INVERTING

    shaping_ratio = shaping_ratio.fillna(1) # NOTE: keep as it is if we can't calculate a ratio (NaN case)
    shaping_ratio_inverted = shaping_ratio_inverted.fillna(1) # NOTE: keep as it is if we can't calculate a ratio (NaN case)

    ret = df_raw[str(exogenous_feature)] * shaping_ratio
    ret_inverted = df_raw[str(exogenous_feature)] * shaping_ratio_inverted

    return {"shaped" : ret, "inverted_shaped": ret_inverted}, exogenous_feature

def pentropy(params):
    '''
    RELATIVE ENTROPY SHAPING
    params = {
        "df_raw" : df_raw,
        "window" : window,
        "exogenous_feature": feature,
        "target_feature": target
    }
    '''
    df_raw = params["df_raw"]
    window = params["window"]
    exogenous_feature = str(params["exogenous_feature"])
    target_feature = str(params["target_feature"])

    target_values, _ = coalesce_series(df_raw[target_feature].values)
    exogenous_values, _ = coalesce_series(df_raw[exogenous_feature].values)

    target_windows = sliding_window_view(target_values, window_shape=window)
    exogenous_windows = sliding_window_view(exogenous_values, window_shape=window)

    result = np.array([relative_entropy(a, b) for a, b in zip(target_windows, exogenous_windows)])
    saliency = pd.Series([np.nan]*(window-1) + list(result))

    shaping_ratio = (saliency-saliency.min())/(saliency.max() - saliency.min()) # normalizing between 0 and 1
    shaping_ratio_inverted = (shaping_ratio - 1).abs() # NOTE: INVERTING

    shaping_ratio = shaping_ratio.fillna(1) # NOTE: keep as it is if we can't calculate a ratio (NaN case)
    shaping_ratio_inverted = shaping_ratio_inverted.fillna(1) # NOTE: keep as it is if we can't calculate a ratio (NaN case)

    ret = df_raw[str(exogenous_feature)] * shaping_ratio
    ret_inverted = df_raw[str(exogenous_feature)] * shaping_ratio_inverted

    return {"shaped" : ret, "inverted_shaped": ret_inverted}, exogenous_feature

def pmutual_info(params):
    '''
    MUTUAL INFORMATION SHAPING
    params = {
        "df_raw" : df_raw,
        "window" : window,
        "exogenous_feature": feature,
        "target_feature": target
    }
    '''
    df_raw = params["df_raw"]
    window = params["window"]
    exogenous_feature = str(params["exogenous_feature"])
    target_feature = str(params["target_feature"])

    target_values, _ = coalesce_series(df_raw[target_feature].values)
    exogenous_values, _ = coalesce_series(df_raw[exogenous_feature].values)

    target_windows = sliding_window_view(target_values, window_shape=window)
    exogenous_windows = sliding_window_view(exogenous_values, window_shape=window)

    result = np.array([mutual_info(a, b) for a, b in zip(target_windows, exogenous_windows)])
    saliency = pd.Series([np.nan]*(window-1) + list(result))

    shaping_ratio = (saliency-saliency.min())/(saliency.max() - saliency.min()) # normalizing between 0 and 1
    shaping_ratio_inverted = (shaping_ratio - 1).abs() # NOTE: INVERTING

    shaping_ratio = shaping_ratio.fillna(1) # NOTE: keep as it is if we can't calculate a ratio (NaN case)
    shaping_ratio_inverted = shaping_ratio_inverted.fillna(1) # NOTE: keep as it is if we can't calculate a ratio (NaN case)

    ret = df_raw[str(exogenous_feature)] * shaping_ratio
    ret_inverted = df_raw[str(exogenous_feature)] * shaping_ratio_inverted

    return {"shaped" : ret, "inverted_shaped": ret_inverted}, exogenous_feature

def pdtw(params):
    '''
    DTW DISTANCE SHAPING
    params = {
        "df_raw" : df_raw,
        "window" : window,
        "exogenous_feature": feature,
        "target_feature": target
    }
    '''
    df_raw = params["df_raw"]
    window = params["window"]
    exogenous_feature = str(params["exogenous_feature"])
    target_feature = str(params["target_feature"])

    target_values = df_raw[target_feature].values
    exogenous_values = df_raw[exogenous_feature].values

    target_windows = sliding_window_view(target_values, window_shape=window)
    exogenous_windows = sliding_window_view(exogenous_values, window_shape=window)

    dtw_dists = []
    for a, b in zip(target_windows, exogenous_windows):
        a = np.array(a, dtype=np.double)
        b = np.array(b, dtype=np.double)
        dtw_dist = dtw.distance_fast(a, b, use_pruning=True)
        dtw_dists += [dtw_dist]

    result = np.array(dtw_dists)
    
    saliency = pd.Series([np.nan]*(window-1) + list(result))

    shaping_ratio = (saliency-saliency.min())/(saliency.max() - saliency.min()) # normalizing between 0 and 1
    shaping_ratio_inverted = (shaping_ratio - 1).abs() # NOTE: INVERTING

    shaping_ratio = shaping_ratio.fillna(1) # NOTE: keep as it is if we can't calculate a ratio (NaN case)
    shaping_ratio_inverted = shaping_ratio_inverted.fillna(1) # NOTE: keep as it is if we can't calculate a ratio (NaN case)

    ret = df_raw[str(exogenous_feature)] * shaping_ratio
    ret_inverted = df_raw[str(exogenous_feature)] * shaping_ratio_inverted

    return {"shaped" : ret, "inverted_shaped": ret_inverted}, exogenous_feature