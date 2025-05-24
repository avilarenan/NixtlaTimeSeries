import logging

from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoMLP, AutoLSTM, AutoNHITS, AutoTFT, AutoNBEATSx, AutoTiDE, AutoTSMixerx, AutoBiTCN, AutoDeepNPTS
from neuralforecast.models import MLP, LSTM, NHITS, TFT, NBEATSx, TiDE, TSMixerx, BiTCN, DeepNPTS

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
    automatic_hyperparam_tuning,
    exog_list=[],
    num_samples=5,
    backend="optuna",
    model_type="univariate"
):

    models_list = [
        AutoTiDE,
        AutoBiTCN,
        # AutoDeepNPTS,
        # AutoLSTM,
        # AutoNHITS,
        # AutoTFT,
        # AutoMLP,
        # AutoNBEATSx
    ] if automatic_hyperparam_tuning else [
        TiDE,
        # BiTCN,
        # DeepNPTS,
        LSTM,
        # NHITS,
        # TFT,
        # MLP,
        # NBEATSx
    ]

    if model_type =="univariate":
        if automatic_hyperparam_tuning:
            models = [
                _model(
                    h=horizon,
                    backend=backend,
                    loss=MSE(),
                    config=general_config(
                        # horizon=horizon,
                        input_size=lookback,
                        exog_list=exog_list,
                        model_name=_model.__name__
                    ),
                    num_samples=num_samples
                ) for _model in models_list
            ]
        else:
            models = [
                get_fixed_hyper_parameter_model(
                    model_name=_model.__name__,
                    horizon=horizon,
                    hist_exog_list=exog_list,
                    lookback=lookback,
                    loss=MSE(),
                ) for _model in models_list
            ]

        return NeuralForecast(models=models, freq=freq)

    elif model_type == "multivariate":
        raise Exception("Not implemented yet")
    else:
        raise Exception(f"Unrecognized model type: {model_type}. Available model types are: univariate and multivariate")