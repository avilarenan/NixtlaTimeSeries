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

from neuralforecast import NeuralForecast

from models import get_nf

HORIZONS = [
    96,
    # 192,
    # 336,
    # 720
]
LOOKBACK = 96
NUM_SAMPLES = 20
ACCURACY_METRICS_TO_EVALUATE = [
    mse,
    # mae,
    # rmse,
    # mape,
    # smape
]


datasets = [
    "ETTh1",
    # "ETTh2",
    # "ETTm1",
    # "ETTm2",
    # "Weather",
    # "ECL",
    # "TrafficL",
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

        evaluation_df = evaluate(cv_df.drop(columns='cutoff'), metrics=ACCURACY_METRICS_TO_EVALUATE)
        evaluation_df['best_model'] = evaluation_df.drop(columns=['metric', 'unique_id']).idxmin(axis=1)

        try:
            Path(f"./results/horizon{horizon}/").mkdir(parents=True, exist_ok=True)
            evaluation_df.to_csv(f"./results/horizon{horizon}/{dataset_name}.csv", index=False)
        except Exception as e:
            print(e)
            evaluation_df.to_csv(f"h{horizon}_{dataset_name}.csv", index=False)


        try:
            Path(f"./saved_models/{dataset_name}_h{horizon}/").mkdir(parents=True, exist_ok=True)
            nf.save(
                path=f"./saved_models/{dataset_name}_h{horizon}/",
                model_index=None,
                overwrite=True,
                save_dataset=True
            )
        except Exception as e:
            print(e)
            nf.save(
                path=f"./saved_models/",
                model_index=None,
                overwrite=True,
                save_dataset=True
            )