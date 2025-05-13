from PyFARM import farm
from scipy.stats import entropy
from fastdtw import fastdtw
from sklearn.feature_selection import mutual_info_regression
from scipy.spatial.distance import euclidean
import numpy as np
import pandas as pd

from numpy.lib.stride_tricks import sliding_window_view

def process_farm(farm_params):
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
    exogenous_feature = farm_params["exogenous_feature"]
    target_feature = farm_params["target_feature"]
    ret = farm(
        refTS=df_raw[target_feature].values,
        qryTS=df_raw[str(exogenous_feature)].values,
        ff_align=False,
        lcwin=window,
        fuzzyc=[1]
    )["qts_shaped"]

    return ret, exogenous_feature

def process_rollcorr(params):
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
    exogenous_feature = params["exogenous_feature"]
    target_feature = params["target_feature"]
    saliency = df_raw[target_feature].rolling(window).corr(df_raw[str(exogenous_feature)])
    
    shaping_ratio = (saliency-saliency.min())/(saliency.max() - saliency.min()) # normalizing between 0 and 1
    ret = df_raw[str(exogenous_feature)][window:] * shaping_ratio

    return ret, exogenous_feature

def process_rollcov(params):
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
    exogenous_feature = params["exogenous_feature"]
    target_feature = params["target_feature"]
    saliency = df_raw[target_feature].rolling(window).cov(df_raw[str(exogenous_feature)])

    shaping_ratio = (saliency-saliency.min())/(saliency.max() - saliency.min()) # normalizing between 0 and 1
    ret = df_raw[str(exogenous_feature)][window:] * shaping_ratio

    return ret, exogenous_feature

def process_entropy(params):
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
    exogenous_feature = params["exogenous_feature"]
    target_feature = params["target_feature"]

    target_windows = sliding_window_view(df_raw[target_feature].values, window_shape=window)
    exogenous_windows = sliding_window_view(df_raw[exogenous_feature].values, window_shape=window)

    result = np.array([entropy(a, b) for a, b in zip(target_windows, exogenous_windows)])
    saliency = pd.Series([np.nan]*(window-1) + list(result))

    shaping_ratio = (saliency-saliency.min())/(saliency.max() - saliency.min()) # normalizing between 0 and 1
    ret = df_raw[str(exogenous_feature)] * shaping_ratio

    return ret, exogenous_feature

def process_mutual_info(params):
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
    exogenous_feature = params["exogenous_feature"]
    target_feature = params["target_feature"]

    target_windows = sliding_window_view(df_raw[target_feature].values, window_shape=window)
    exogenous_windows = sliding_window_view(df_raw[exogenous_feature].values, window_shape=window)

    result = np.array([mutual_info_regression(a.reshape(-1, 1), b.reshape(-1, 1)) for a, b in zip(target_windows, exogenous_windows)])
    saliency = pd.Series([np.nan]*(window-1) + list(result))

    shaping_ratio = (saliency-saliency.min())/(saliency.max() - saliency.min()) # normalizing between 0 and 1
    ret = df_raw[str(exogenous_feature)] * shaping_ratio

    return ret, exogenous_feature

def process_dtw(params):
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
    exogenous_feature = params["exogenous_feature"]
    target_feature = params["target_feature"]

    def fastdtw_dist(x, y):
        distance, path = fastdtw(x, y, dist=euclidean)
        return distance

    df_raw = params["df_raw"]
    window = params["window"]
    exogenous_feature = params["exogenous_feature"]
    target_feature = params["target_feature"]

    target_windows = sliding_window_view(df_raw[target_feature].values, window_shape=window)
    exogenous_windows = sliding_window_view(df_raw[exogenous_feature].values, window_shape=window)

    result = np.array([fastdtw_dist(a.reshape(-1, 1), b.reshape(-1, 1)) for a, b in zip(target_windows, exogenous_windows)])
    saliency = pd.Series([np.nan]*(window-1) + list(result))

    shaping_ratio = (saliency-saliency.min())/(saliency.max() - saliency.min()) # normalizing between 0 and 1
    ret = df_raw[str(exogenous_feature)] * shaping_ratio

    return ret, exogenous_feature