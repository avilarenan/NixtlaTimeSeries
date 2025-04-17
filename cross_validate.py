import logging
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)

import warnings
warnings.filterwarnings('ignore')

from datasets_metadata import ts_metadata
import pandas as pd

from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mse, mae, rmse, mape

from models import get_nf

dataset_name = "ETTm2"
exog_list = ts_metadata[dataset_name]["exog_list"]
target_ts = ts_metadata[dataset_name]["target_ts"]
freq = ts_metadata[dataset_name]["freq"]

df = pd.read_csv(f"./processed_data/{dataset_name}.csv")
df["ds"] = pd.to_datetime(df["ds"])

# nf = NeuralForecast.load(path=f'./saved_models/{dataset_name}')

horizon = 24

nf = get_nf(
    horizon=horizon,
    freq=freq,
    exog_list=exog_list,
    num_samples=20,
    backend="optuna"
)

# cv_df = nf.cross_validation(df, n_windows=10, step_size=100)

cv_df = nf.cross_validation(df, val_size=1000, test_size=2000, n_windows=None)

cv_df.columns = cv_df.columns.str.replace('-median', '')

evaluation_df = evaluate(cv_df.drop(columns='cutoff'), metrics=[mse, mae, rmse, mape])
evaluation_df['best_model'] = evaluation_df.drop(columns=['metric', 'unique_id']).idxmin(axis=1)
evaluation_df.to_csv(f"./results/{dataset_name}")

nf.save(
    path=f"./saved_models/{dataset_name}.csv",
    model_index=None, 
    overwrite=True,
    save_dataset=True
)