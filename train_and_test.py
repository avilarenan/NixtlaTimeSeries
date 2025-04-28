import logging
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)

import warnings
warnings.filterwarnings('ignore')
import os

from datasets_metadata import ts_metadata
import pandas as pd

from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mse, mae, rmse, mape, smape

from neuralforecast import NeuralForecast

from models import get_nf

datasets = [
    # "ETTh1", 
    # "ETTh2",
    "ETTm1",
    "ETTm2"
]

for dataset_name in datasets:
    print(f"Running dataset {dataset_name}")
    exog_list = ts_metadata[dataset_name]["exog_list"]
    target_ts = ts_metadata[dataset_name]["target_ts"]
    freq = ts_metadata[dataset_name]["freq"]

    test_size = ts_metadata[dataset_name]["test_size"]
    valid_size = ts_metadata[dataset_name]["valid_size"]

    df = pd.read_csv(f"./processed_data/{dataset_name}.csv")
    df["ds"] = pd.to_datetime(df["ds"])

    # nf = NeuralForecast.load(path=f'./saved_models/{dataset_name}')

    horizon = 96
    lookback = 96

    nf = get_nf(
        horizon=horizon,
        lookback=lookback,
        freq=freq,
        exog_list=exog_list,
        num_samples=20,
        backend="optuna"
    )

    dataset_path = f"./processed_data/train/{dataset_name}"
    files_list = [f"{dataset_path}/{dir}" for dir in os.listdir(dataset_path)]

    nf.fit(df=files_list, val_size=valid_size)

    nf.save(
        path=f"./saved_models/{dataset_name}_h{horizon}.csv",
        model_index=None,
        overwrite=True,
        save_dataset=False
    )

    # nf = NeuralForecast.load(path=f'./saved_models/{dataset_name}')

    test_df = pd.read_csv(f"./processed_data/test/{dataset_name}.csv")
    test_df["ds"] = pd.to_datetime(test_df["ds"])
    
    cv_df = nf.cross_validation(df=test_df, val_size=0, test_size=len(test_df), step_size=1, n_windows=None, prevent_retraining=True)

    cv_df.columns = cv_df.columns.str.replace('-median', '')

    evaluation_df = evaluate(cv_df.drop(columns='cutoff'), metrics=[mse, mae, rmse, mape, smape])
    evaluation_df['best_model'] = evaluation_df.drop(columns=['metric', 'unique_id']).idxmin(axis=1)
    evaluation_df.to_csv(f"./results/{dataset_name}")

    nf.save(
        path=f"./saved_models/{dataset_name}_h{horizon}.csv",
        model_index=None,
        overwrite=True,
        save_dataset=True
    )

