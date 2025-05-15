import logging

from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoMLP, AutoLSTM, AutoNHITS, AutoTFT, AutoNBEATSx, AutoTiDE, AutoTSMixerx, AutoBiTCN, AutoDeepNPTS

logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)

import warnings
warnings.filterwarnings('ignore')

from model_metadata import general_config

from ray import tune

from neuralforecast.losses.pytorch import MAE, MSE, RMSE, MAPE, SMAPE

def get_nf(horizon, lookback, freq, exog_list=[], num_samples=5, backend="optuna", model_type="univariate"):

    if model_type =="univariate":
        models = [
            _model(
                h=horizon,
                backend=backend,
                loss=MAE(),
                config=general_config(
                    # horizon=horizon,
                    input_size=lookback,
                    exog_list=exog_list,
                    model_name=_model.__name__
                ),
                num_samples=num_samples
            ) for _model in [AutoTiDE, AutoBiTCN, AutoDeepNPTS, AutoLSTM, AutoNHITS, AutoTFT, AutoMLP, AutoNBEATSx]
        ]

        return NeuralForecast(models=models, freq=freq)

    elif model_type == "multivariate":
        raise Exception("Not implemented yet")
    else:
        raise Exception(f"Unrecognized model type: {model_type}. Available model types are: univariate and multivariate")