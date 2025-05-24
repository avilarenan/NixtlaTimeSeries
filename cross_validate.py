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
from neuralforecast import NeuralForecast

from models import get_nf

HORIZONS = [
    96,
    # 192,
    # 336,
    # 720
]
LOOKBACK = 96
NUM_SAMPLES = 2
INPUT_DATA_FORMAT = ".csv"
INPUT_DATA_PATH = "./processed_data" # NOTE: without slash in the end
OUTPUT_RESULTS_PATH = "./results" # NOTE: without slash in the end
AUTOMATIC_HYPERPARAM_TUNING = True
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

for horizon in tqdm(HORIZONS):
    for dataset_name in tqdm(DATASETS):
        print(f"Running dataset {dataset_name} | Horizon: {horizon}")

        exog_list = ts_metadata[dataset_name]["exog_list"]
        target_ts = ts_metadata[dataset_name]["target_ts"]
        freq = ts_metadata[dataset_name]["freq"]
        test_size = ts_metadata[dataset_name]["test_size"]
        valid_size = ts_metadata[dataset_name]["valid_size"]

        df = read_df_from_file(path=INPUT_DATA_PATH, filename=dataset_name, format=INPUT_DATA_FORMAT)

        df["ds"] = pd.to_datetime(df["ds"])

        nf = get_nf(
            horizon=horizon,
            lookback=LOOKBACK,
            freq=freq,
            automatic_hyperparam_tuning=AUTOMATIC_HYPERPARAM_TUNING,
            exog_list=exog_list,
            num_samples=NUM_SAMPLES,
            backend="optuna"
        )

        cv_df = nf.cross_validation(df=df, val_size=valid_size, test_size=test_size, step_size=1, n_windows=None, verbose=True)

        cv_df.columns = cv_df.columns.str.replace('-median', '')

        evaluation_df = evaluate(cv_df.drop(columns='cutoff'), metrics=ACCURACY_METRICS_TO_EVALUATE)
        evaluation_df['best_model'] = evaluation_df.drop(columns=['metric', 'unique_id']).idxmin(axis=1)

        if AUTOMATIC_HYPERPARAM_TUNING:
            for model in nf.models:
                trials_df = model.results.trials_dataframe()
                save_df_to_file(df=trials_df, path=f"{OUTPUT_RESULTS_PATH}/horizon{horizon}", filename=f"{model}_trials", format=".csv")
        else:
            base_folder = "saved_models_auto" if AUTOMATIC_HYPERPARAM_TUNING else "saved_models"
            Path(f"./{base_folder}/{dataset_name}_h{horizon}/").mkdir(parents=True, exist_ok=True)
            nf.save(
                path=f"./{base_folder}/{dataset_name}_h{horizon}/",
                model_index=None,
                overwrite=True,
                save_dataset=True
            )

        save_df_to_file(df=evaluation_df, path=f"{OUTPUT_RESULTS_PATH}/horizon{horizon}", filename=f"{dataset_name}_metrics", format=".csv")
        save_df_to_file(df=cv_df, path=f"{OUTPUT_RESULTS_PATH}/horizon{horizon}", filename=f"{dataset_name}_pred", format=".parquet")