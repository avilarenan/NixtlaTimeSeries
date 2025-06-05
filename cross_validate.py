# NOTE: this script should be run from within root repo directory

import logging
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
from tqdm import tqdm
import math
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

from datasets_metadata import ts_metadata
import pandas as pd

from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mse, mae, rmse, mape, smape
from preprocessing_utilities import read_df_from_file, save_df_to_file

from neuralforecast.auto import AutoMLP, AutoLSTM, AutoNHITS, AutoTFT, AutoNBEATSx, AutoTiDE, AutoTSMixerx, AutoBiTCN, AutoDeepNPTS, AutoGRU, AutoTCN
from neuralforecast.models import MLP, LSTM, NHITS, TFT, NBEATSx, TiDE, TSMixerx, BiTCN, DeepNPTS, GRU, TCN

from models import get_nf

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('experiment.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

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
    "ipfarm",
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
logger.info(f"model_names_list: {MODEL_NAMES_LIST}")

STATE_FILE = "exps_state.csv" # NOTE: assuming it is in root repo folder 
if AUTOMATIC_HYPERPARAM_TUNING:
    STATE_FILE = "auto_exps_state.csv"
    SHAPING_PROCESSES = ["identity"]
    SHAPING_WINDOWS = ["N/A"]

try:
    df_exps_state = pd.read_csv(STATE_FILE, index_col=[0,1], header=[0,1,2]) # NOTE: restablishing multiindex
except FileNotFoundError as e:
    logger.info("Experiments state file not found, creating a new one from scratch.")
    STR_HORIZONS = [str(h) for h in HORIZONS]
    STR_SHAPING_WINDOWS = [str(w) for w in SHAPING_WINDOWS] # NOTE: setting as string because reading and writing multiindex dfs to csv and correctly recognizing data headers and indexes types is problematic

    row_levels = [STR_HORIZONS, MODEL_NAMES_LIST]
    col_levels = [DATASETS, SHAPING_PROCESSES, STR_SHAPING_WINDOWS]

    row_index = pd.MultiIndex.from_product(row_levels, names=['Horizons', 'Models'])
    col_index = pd.MultiIndex.from_product(col_levels, names=['Datasets', 'Shaping Process', 'Shaping Window'])

    df_exps_state = pd.DataFrame(index=row_index, columns=col_index)
    df_exps_state = df_exps_state.fillna(True)

    for dset in DATASETS:
        for sh_window in STR_SHAPING_WINDOWS:
            df_exps_state.drop((dset, "identity", sh_window), axis=1, inplace=True)

        df_exps_state[(dset, "identity", "N/A")] = False

    df_exps_state.to_csv(STATE_FILE)

for horizon in tqdm(HORIZONS):
    logger.info(f"Running horizon: {horizon}")
    for dataset_name in tqdm(DATASETS):
        try:
            logger.info(f"Running dataset: {dataset_name}")

            exog_list = ts_metadata[dataset_name]["exog_list"]
            target_ts = ts_metadata[dataset_name]["target_ts"]
            freq = ts_metadata[dataset_name]["freq"]
            test_size = ts_metadata[dataset_name]["test_size"]
            valid_size = ts_metadata[dataset_name]["valid_size"]

            for shaping_process in tqdm(SHAPING_PROCESSES):
                logger.info(f"Running shaping_process: {shaping_process}")
                _shaping_windows = SHAPING_WINDOWS if shaping_process != "identity" else ["N/A"]
                for shaping_window in tqdm(_shaping_windows):
                    try:
                        logger.info(f"Running shaping_window: {shaping_window}")
                        if shaping_process == "identity" and shaping_window == "N/A": # NOTE: handling baseline non processed case
                            shaped_dataset_name = f"{dataset_name}_na_{shaping_process}"
                        else:
                            shaped_dataset_name = f"{dataset_name}_w{shaping_window}_{shaping_process}"
                        
                        logger.info(f"Running shaped_dataset_name: {shaped_dataset_name}")

                        df = read_df_from_file(
                            path=INPUT_DATA_PATH,
                            filename=shaped_dataset_name,
                            format=INPUT_DATA_FORMAT
                        )

                        df["ds"] = pd.to_datetime(df["ds"])

                        for model in MODELS_LIST:
                            _model_name = find_model_name(model)
                            logger.info(f"Running model: {_model_name}")
                            logger.info(f"state cell [{[horizon, _model_name]}|{[dataset_name, shaping_process, str(shaping_window)]}] = {df_exps_state.loc[(horizon, _model_name), (dataset_name, shaping_process, str(shaping_window))]}")
                            if df_exps_state.loc[(horizon, _model_name), (dataset_name, shaping_process, str(shaping_window))] == True: # WARNING: for some reason whe reading the csv from auto, shaping window is read as int and from normal it is read as str
                                logger.info(f"Skipping {_model_name} because it is done in experiments states file.")
                                continue # NOTE: Skip if already ran
                            nf = get_nf(
                                horizon=horizon,
                                lookback=LOOKBACK,
                                freq=freq,
                                models_list=[model],
                                automatic_hyperparam_tuning=AUTOMATIC_HYPERPARAM_TUNING,
                                dataset_name=dataset_name,
                                exog_list=exog_list,
                                num_samples=NUM_SAMPLES,
                                backend="optuna",
                                use_best_of_trials=USE_BEST_HYPERPARAMETERS_OF_TRIALS
                            )

                            cv_df = nf.cross_validation(df=df, val_size=valid_size, test_size=test_size, step_size=1, n_windows=None, verbose=True)

                            cv_df.columns = cv_df.columns.str.replace('-median', '')

                            evaluation_df = evaluate(cv_df.drop(columns='cutoff'), metrics=ACCURACY_METRICS_TO_EVALUATE)

                            auto_label_str = "_auto" if AUTOMATIC_HYPERPARAM_TUNING else "" 

                            if AUTOMATIC_HYPERPARAM_TUNING:
                                auto_label_str = "_auto"
                                for model in nf.models:
                                    trials_df = model.results.trials_dataframe()
                                    save_df_to_file(df=trials_df, path=f"{OUTPUT_RESULTS_PATH}/{shaped_dataset_name}/{model}_horizon{horizon}{auto_label_str}", filename=f"{model}_trials", format=".csv")
                            else:
                                auto_label_str = ""
                                base_folder = "saved_models"
                                model_class_name = ""
                                if model.__class__.__name__ == "type":
                                    model_class_name = model.__name__
                                else:
                                    model_class_name = model.__class__.__name__
                                
                                print(f"\n\n\n*****{model_class_name}\n\n\n")
                                
                                Path(f"./{base_folder}/{model_class_name}_{shaped_dataset_name}_h{horizon}/").mkdir(parents=True, exist_ok=True)
                                nf.save(
                                    path=f"./{base_folder}/{model_class_name}_{shaped_dataset_name}_h{horizon}/",
                                    model_index=None,
                                    overwrite=True,
                                    save_dataset=True
                                )
                        
                                save_df_to_file(df=evaluation_df, path=f"{OUTPUT_RESULTS_PATH}/{shaped_dataset_name}/{model_class_name}_horizon{horizon}{auto_label_str}", filename=f"metrics", format=".csv")
                                save_df_to_file(df=cv_df, path=f"{OUTPUT_RESULTS_PATH}/{shaped_dataset_name}/{model_class_name}_horizon{horizon}{auto_label_str}", filename=f"pred", format=".parquet")
                            
                            df_exps_state.loc[(horizon, _model_name), (dataset_name, shaping_process, str(shaping_window))] = True # Register experiment state run
                            df_exps_state.to_csv(STATE_FILE)
                    except Exception as e:
                        error_msg = f"Error (inner case) when running: horizon: {horizon} | dataset_name: {dataset_name} | shaping_process: {shaping_process} | shaping_window: {shaping_window} | shaped_dataset_name: {shaped_dataset_name} | model : {model}"
                        logger.error(error_msg)
                        logger.exception(error_msg)
        except Exception as e:
            error_msg = f"Error (outer case) when running: horizon: {horizon} | dataset_name: {dataset_name} | shaping_process: {shaping_process} | shaping_window: {shaping_window} | shaped_dataset_name: {shaped_dataset_name} | model : {model}"
            logger.error(error_msg)
            logger.exception(error_msg)
