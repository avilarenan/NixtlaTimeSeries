from utilsforecast.losses import mse, mae, rmse, mape, smape
from neuralforecast.auto import AutoMLP, AutoLSTM, AutoNHITS, AutoTFT, AutoNBEATSx, AutoTiDE, AutoTSMixerx, AutoBiTCN, AutoDeepNPTS, AutoGRU, AutoTCN
from neuralforecast.models import MLP, LSTM, NHITS, TFT, NBEATSx, TiDE, TSMixerx, BiTCN, DeepNPTS, GRU, TCN

HORIZONS = [
    96,
    192,
    336,
    720
]

LOOKBACK = 96
NUM_SAMPLES = 20
INPUT_DATA_FORMAT = ".csv"
INPUT_DATA_PATH = "./processed_data" # NOTE: without slash in the end
OUTPUT_RESULTS_PATH = "./results" # NOTE: without slash in the end
AUTOMATIC_HYPERPARAM_TUNING = False
USE_BEST_HYPERPARAMETERS_OF_TRIALS = True

if AUTOMATIC_HYPERPARAM_TUNING:
    USE_BEST_HYPERPARAMETERS_OF_TRIALS = False

ACCURACY_METRICS_TO_EVALUATE = [
    mse,
    mae,
    rmse,
    mape,
    smape
]

DATASETS = [
    "ETTh1",
    # "ETTh2",
    # "ETTm1",
    # "ETTm2",
    # "Weather",
    # "ECL",
    # "TrafficL",
]

SHAPING_PROCESSES = [ # TODO: programmatically sync with process function names
    "identity",
    "pfarm",
    "prollcorr",
    "prollcov",
    "pentropy",
    "pmutual_info",
    "iprollcorr",
    "iprollcov",
    "ipentropy",
    "ipmutual_info",
    # "dtw"
]

SHAPING_WINDOWS = [ # TODO: programmatically sync with process shaping windows
    501,
    751,
    1001,
    1251,
    1501
]

MODELS_LIST = [
    AutoTFT,
    AutoTSMixerx,
    AutoGRU,
    AutoTCN,
    AutoTiDE,
    AutoBiTCN,
    AutoDeepNPTS,
    AutoLSTM,
    AutoNHITS,
    AutoMLP,
    AutoNBEATSx,
] if AUTOMATIC_HYPERPARAM_TUNING else [
    TFT,
    TSMixerx,
    GRU,
    TCN,
    TiDE,
    BiTCN,
    DeepNPTS,
    LSTM,
    NHITS,
    MLP,
    NBEATSx
]

def find_model_name(model_class):
    if model_class.__class__.__name__ == "type":
        return model_class.__name__
    else:
        return model_class.__class__.__name__
MODEL_NAMES_LIST = [find_model_name(model) for model in MODELS_LIST]

STATE_FILE = "exps_state.csv" # NOTE: assuming it is in root repo folder 
if AUTOMATIC_HYPERPARAM_TUNING:
    STATE_FILE = "auto_exps_state.csv"
    SHAPING_PROCESSES = ["identity"]
    SHAPING_WINDOWS = ["N/A"]
