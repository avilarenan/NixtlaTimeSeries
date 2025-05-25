import logging
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
from tqdm import tqdm

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

HORIZONS = [
    # 96,
    192,
    336,
    720
]
LOOKBACK = 96
NUM_SAMPLES = 10
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
    # rmse,
    # mape,
    # smape
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

for horizon in tqdm(HORIZONS):
    for dataset_name in tqdm(DATASETS):

        exog_list = ts_metadata[dataset_name]["exog_list"]
        target_ts = ts_metadata[dataset_name]["target_ts"]
        freq = ts_metadata[dataset_name]["freq"]
        test_size = ts_metadata[dataset_name]["test_size"]
        valid_size = ts_metadata[dataset_name]["valid_size"]

        df = read_df_from_file(path=INPUT_DATA_PATH, filename=dataset_name, format=INPUT_DATA_FORMAT)

        df["ds"] = pd.to_datetime(df["ds"])

        for model in MODELS_LIST:
            print(f"Running dataset {dataset_name} | Horizon: {horizon} | Model: {model}")
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
                    save_df_to_file(df=trials_df, path=f"{OUTPUT_RESULTS_PATH}/{dataset_name}/{model}_horizon{horizon}{auto_label_str}", filename=f"{model}_trials", format=".csv")
            else:
                auto_label_str = ""
                base_folder = "saved_models"
                Path(f"./{base_folder}/{model}_{dataset_name}_h{horizon}/").mkdir(parents=True, exist_ok=True)
                nf.save(
                    path=f"./{base_folder}/{model}_{dataset_name}_h{horizon}/",
                    model_index=None,
                    overwrite=True,
                    save_dataset=True
                )
            
            save_df_to_file(df=evaluation_df, path=f"{OUTPUT_RESULTS_PATH}/{dataset_name}/{model.__name__}_horizon{horizon}{auto_label_str}", filename=f"metrics", format=".csv")
            save_df_to_file(df=cv_df, path=f"{OUTPUT_RESULTS_PATH}/{dataset_name}/{model.__name__}_horizon{horizon}{auto_label_str}", filename=f"pred", format=".parquet")