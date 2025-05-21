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
NUM_SAMPLES = 50
AUTOMATIC_HYPERPARAM_TUNING = False
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

        df_train = pd.read_parquet(f"./processed_data/{dataset_name}_train.parquet")
        df_test = pd.read_parquet(f"./processed_data/{dataset_name}_test.parquet")
        
        df_train["ds"] = pd.to_datetime(df_train["ds"])
        df_test["ds"] = pd.to_datetime(df_test["ds"])

        nf = get_nf(
            horizon=horizon,
            lookback=LOOKBACK,
            freq=freq,
            automatic_hyperparam_tuning=AUTOMATIC_HYPERPARAM_TUNING,
            exog_list=exog_list,
            num_samples=NUM_SAMPLES,
            backend="optuna"
        )

        nf.fit(
            df=df_train,
            val_size=valid_size,
            verbose=True
        )

        fcst_df = nf.predict(
            df=df_test,
        )

        fcst_df.columns = fcst_df.columns.str.replace('-median', '')

        print(fcst_df)

        evaluation_df = evaluate(
            fcst_df.merge(df_test[["unique_id", "ds", "y"]],on=['unique_id', 'ds']),
            metrics=ACCURACY_METRICS_TO_EVALUATE,
            agg_fn='mean'
        )

        print(evaluation_df)

        print(df_test)

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