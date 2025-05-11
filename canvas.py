from datasetsforecast.long_horizon2 import LongHorizon2
# from datasetsforecast.long_horizon import LongHorizon
import pandas as pd
pd.options.plotting.backend = "plotly"
import warnings
warnings.filterwarnings('ignore')
from PyFARM import farm
from tqdm.notebook import tqdm

from datasets_metadata import ts_metadata

# print(pd.read_parquet("./processed_data/Weather.parquet").columns)

dataset_name = "ECL"

Y_df = LongHorizon2.load(directory='data', group=dataset_name)

target_ts = ts_metadata[dataset_name]["target_ts"]
exog_list = ts_metadata[dataset_name]["exog_list"]
test_size = ts_metadata[dataset_name]["test_size"]
valid_size = ts_metadata[dataset_name]["valid_size"]
farm_windows = ts_metadata[dataset_name]["farm_windows"]

Y_df["ds"] = pd.to_datetime(Y_df["ds"])
Y_df = Y_df.set_index("ds")
Y_df["unique_id"] = Y_df["unique_id"].astype('string')
print(Y_df.dtypes)
print(Y_df)
print(Y_df["unique_id"].unique())

# Pivotting original dataset in order to consider other series as exogenous
df_raw = Y_df[Y_df["unique_id"] == target_ts]
for item in tqdm(Y_df["unique_id"].unique()):
    item = str(item)
    if item == target_ts:
        continue
    df_raw[item] = Y_df[Y_df["unique_id"] == item]["y"]
df_raw = df_raw.reset_index()
df_raw = df_raw.drop("index", axis=1)
df_raw["unique_id"] = f"{target_ts}_raw"

print(df_raw)