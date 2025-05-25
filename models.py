import logging

from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoMLP, AutoLSTM, AutoNHITS, AutoTFT, AutoNBEATSx, AutoTiDE, AutoTSMixerx, AutoBiTCN, AutoDeepNPTS, AutoGRU, AutoTCN
from neuralforecast.models import MLP, LSTM, NHITS, TFT, NBEATSx, TiDE, TSMixerx, BiTCN, DeepNPTS, GRU, TCN

logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)

import warnings
warnings.filterwarnings('ignore')

from model_metadata import general_config, get_fixed_hyper_parameter_model

from ray import tune

from neuralforecast.losses.pytorch import MAE, MSE, RMSE, MAPE, SMAPE

def get_nf(
    horizon,
    lookback,
    freq,
    models_list,
    automatic_hyperparam_tuning,
    dataset_name,
    exog_list=[],
    num_samples=5,
    backend="optuna",
    use_best_of_trials=False
):

    if automatic_hyperparam_tuning:

        models = []

        for _model in models_list:
            kwargs = {
                "h": horizon,
                "backend": backend,
                "loss": MSE(),
                "num_samples": num_samples
            }
            if _model.__name__ in ["AutoTSMixerx"]:
                kwargs["n_series"] = 1
            if _model.__name__ in ["AutoGRU", "AutoLSTM", "AutoRNN"]:
                lookback = horizon # NOTE: It seems recursive models need at least the same amount of data in the input_size as in horizon

            kwargs["config"] = general_config(
                horizon=horizon,
                input_size=lookback,
                exog_list=exog_list,
                model_name=_model.__name__
            )

            models += [_model(**kwargs)]
        
    else:
        models = [
            get_fixed_hyper_parameter_model(
                model_name=_model.__name__,
                horizon=horizon,
                hist_exog_list=exog_list,
                lookback=lookback,
                loss=MSE(),
                dataset_name=dataset_name,
                use_best_of_trials=use_best_of_trials
            ) for _model in models_list
        ]

    return NeuralForecast(models=models, freq=freq)