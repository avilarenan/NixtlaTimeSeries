import logging
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')
import os

from datasets_metadata import ts_metadata
import pandas as pd

from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mse, mae, rmse, mape, smape

from neuralforecast import NeuralForecast

from models import get_nf

HORIZONS = [96, 192, 336, 720]
LOOKBACK = 96

NUM_SAMPLES = 20

datasets = [
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "Weather",
    "ECL",
    "TrafficL",
]

for horizon in tqdm(HORIZONS):
    for dataset_name in tqdm(datasets):
        print(f"Running dataset {dataset_name} | Horizon: {horizon}")
        exog_list = ts_metadata[dataset_name]["exog_list"]
        target_ts = ts_metadata[dataset_name]["target_ts"]
        freq = ts_metadata[dataset_name]["freq"]

        test_size = ts_metadata[dataset_name]["test_size"]
        valid_size = ts_metadata[dataset_name]["valid_size"]

        df = pd.read_parquet(f"./processed_data/{dataset_name}.parquet")
        # df = pd.read_csv(f"./processed_data/{dataset_name}.csv")
        df["ds"] = pd.to_datetime(df["ds"])

        # nf = NeuralForecast.load(path=f'./saved_models/{dataset_name}')

        nf = get_nf(
            horizon=horizon,
            lookback=LOOKBACK,
            freq=freq,
            exog_list=exog_list,
            num_samples=NUM_SAMPLES,
            backend="optuna"
        )

        cv_df = nf.cross_validation(df=df, val_size=valid_size, test_size=test_size, step_size=1, n_windows=None, verbose=True)

        cv_df.columns = cv_df.columns.str.replace('-median', '')

        evaluation_df = evaluate(cv_df.drop(columns='cutoff'), metrics=[mse, mae, rmse, mape, smape])
        evaluation_df['best_model'] = evaluation_df.drop(columns=['metric', 'unique_id']).idxmin(axis=1)
        evaluation_df.to_csv(f"./results/horizon{horizon}/{dataset_name}.csv", index=False)

        nf.save(
            path=f"./saved_models/{dataset_name}_h{horizon}.csv",
            model_index=None,
            overwrite=True,
            save_dataset=True
        )