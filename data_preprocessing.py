import warnings
from datasetsforecast.long_horizon2 import LongHorizon2
# from datasetsforecast.long_horizon import LongHorizon
import pandas as pd
pd.options.plotting.backend = "plotly"
from tqdm.notebook import tqdm
from joblib import Parallel, delayed
import joblib
from datasets_metadata import ts_metadata
import contextlib
from preprocessing_utilities import process_farm, process_rollcorr, process_rollcov, process_entropy, process_mutual_info, process_dtw
warnings.filterwarnings('ignore')

PLOT = False
PARALLEL = True
datasets_names = [
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "Weather",
    "ECL",
    "TrafficL"
]

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

for dataset_name in tqdm(datasets_names, desc="Datasets"):
    Y_df = LongHorizon2.load(directory='data', group=dataset_name)
    Y_df.columns = [str(col) for col in Y_df.columns]

    target_ts = ts_metadata[dataset_name]["target_ts"]
    exog_list = ts_metadata[dataset_name]["exog_list"]
    test_size = ts_metadata[dataset_name]["test_size"]
    valid_size = ts_metadata[dataset_name]["valid_size"]
    farm_windows = ts_metadata[dataset_name]["farm_windows"]

    Y_df["ds"] = pd.to_datetime(Y_df["ds"])
    Y_df = Y_df.set_index("ds")

    # Pivotting original dataset in order to consider other series as exogenous
    df_raw = Y_df[Y_df["unique_id"] == target_ts]
    for new_column_name, group in Y_df.groupby("unique_id"):
        if new_column_name == target_ts:
            continue
        df_raw[new_column_name] = group["y"].values
        df_raw[new_column_name] = df_raw[new_column_name].astype(float)


    df_raw = df_raw.reset_index()
    df_raw = df_raw.drop("index", axis=1)
    df_raw["unique_id"] = f"{target_ts}_raw"

    df_raw.columns = [str(col) for col in df_raw.columns]

    # # SHAPING
    ref_ts = "y"

    list_of_process_fns = [
        process_farm,
        process_rollcorr,
        process_rollcov,
        process_entropy,
        process_mutual_info,
        process_dtw,
    ]
    
    list_of_processed_dfs = []
    for window in tqdm(farm_windows, leave=False, desc="Iterating windows"):
        
        df_roll_corr = df_raw.copy()

        rolling_stats_params_list = []
        for feature in exog_list:
            rolling_stats_params_list += [
                {
                    "df_raw": df_raw,
                    "window": window,
                    "target_feature": ref_ts,
                    "exogenous_feature": feature
                }
            ]

        for process_fn in tqdm(list_of_process_fns, leave=False, desc="Prcessing fns"):
            
            if PARALLEL:
                tqdm_it = tqdm(desc="Parallel Feature engineering processing", total=len(rolling_stats_params_list), leave=False)
                with tqdm_joblib(tqdm_it) as progress_bar:
                    results_processed = Parallel(n_jobs=-1)(delayed(process_fn)(param) for param in rolling_stats_params_list)
                tqdm_it.container.close()
            else:
                # SEQUENTIAL PROCESSING
                results_processed = []
                for param in tqdm(rolling_stats_params_list, desc="Sequential Feature engineering processing", leave=False):
                    results_processed += [process_fn(param)]
                
            df_processed = df_raw.copy()
            df_processed_inverted = df_raw.copy()
            
            skip_inverted = False

            for agg_qts_shaped, feature in results_processed:
                if feature == ref_ts:
                    continue

                qts_shaped = agg_qts_shaped.get("shaped")
                if qts_shaped is None:
                    raise Exception(f"No shaped ts provided: {qts_shaped}")
                else:
                    df_processed[str(feature)] = qts_shaped
                
                qts_shaped_inverted = agg_qts_shaped.get("inverted_shaped")
                if qts_shaped_inverted is None:
                    skip_inverted = True
                else:
                    df_processed_inverted[str(feature)] = qts_shaped_inverted

            df_processed["unique_id"] = f"{target_ts}_w{window}_exogenous_{process_fn.__name__}"
            list_of_processed_dfs += [df_processed]

            if not skip_inverted:
                df_processed_inverted["unique_id"] = f"{target_ts}_w{window}_exogenous_{process_fn.__name__}_inverted"
                list_of_processed_dfs += [df_processed_inverted]
        
    df_processed = pd.concat(list_of_processed_dfs)

    df = pd.concat([df_raw, df_processed])
    # df.to_csv(f"./processed_data/{dataset_name}.csv", index=False)
    df.to_parquet(f"./processed_data/{dataset_name}.parquet", index=False)