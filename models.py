import logging

from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoMLP, AutoLSTM, AutoNHITS, AutoTFT, AutoNBEATSx, AutoTiDE, AutoTSMixerx, AutoBiTCN, AutoDeepNPTS

logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)

import warnings
warnings.filterwarnings('ignore')

from model_metadata import config_model_generator

def get_nf(horizon, freq, exog_list=[], num_samples=5, backend="optuna", model_type="univariate"):


    if model_type =="univariate":
        models = [
            AutoLSTM(
                    h=horizon,
                    backend=backend,
                    config=config_model_generator(
                        horizon=horizon,
                        model_class=AutoLSTM,
                        additional_options={
                            "hist_exog_list": exog_list,
                            "early_stop_patience_steps": 15,
                            "val_check_steps": 1
                        },
                        backend=backend
                    ),
                    num_samples=num_samples
            ),
            AutoNHITS(
                    h=horizon,
                    backend=backend,
                    config=config_model_generator(
                        horizon=horizon,
                        model_class=AutoNHITS,
                        additional_options={
                            "hist_exog_list": exog_list,
                            "early_stop_patience_steps": 15,
                            "val_check_steps": 1
                        },
                        backend=backend
                    ),
                    num_samples=num_samples
            ),
            AutoTFT(
                    h=horizon,
                    backend=backend,
                    config=config_model_generator(
                        horizon=horizon,
                        model_class=AutoTFT,
                        additional_options={
                            "hist_exog_list": exog_list,
                            "early_stop_patience_steps": 15,
                            "val_check_steps": 1
                        },
                        backend=backend
                    ),
                    num_samples=num_samples
            ),
            AutoMLP(
                    h=horizon,
                    backend=backend,
                    config=config_model_generator(
                        horizon=horizon,
                        model_class=AutoMLP,
                        additional_options={
                            "hist_exog_list": exog_list,
                            "early_stop_patience_steps": 15,
                            "val_check_steps": 1
                        },
                        backend=backend
                    ),
                    num_samples=num_samples
            ),
            AutoNBEATSx(
                    h=horizon,
                    backend=backend,
                    config=config_model_generator(
                        horizon=horizon,
                        model_class=AutoNBEATSx,
                        additional_options={
                            "hist_exog_list": exog_list,
                            "early_stop_patience_steps": 15,
                            "val_check_steps": 1
                        },
                        backend=backend
                    ),
                    num_samples=num_samples
            ),
            AutoTiDE(
                    h=horizon,
                    backend=backend,
                    config=config_model_generator(
                        horizon=horizon,
                        model_class=AutoTiDE,
                        additional_options={
                            "hist_exog_list": exog_list,
                            "early_stop_patience_steps": 15,
                            "val_check_steps": 1
                        },
                        backend=backend
                    ),
                    num_samples=num_samples
            ),
            AutoBiTCN(
                    h=horizon,
                    backend=backend,
                    config=config_model_generator(
                        horizon=horizon,
                        model_class=AutoBiTCN,
                        additional_options={
                            "hist_exog_list": exog_list,
                            "early_stop_patience_steps": 15,
                            "val_check_steps": 1
                        },
                        backend=backend
                    ),
                    num_samples=num_samples
            ),
            AutoDeepNPTS(
                    h=horizon,
                    backend=backend,
                    config=config_model_generator(
                        horizon=horizon,
                        model_class=AutoDeepNPTS,
                        additional_options={
                            "hist_exog_list": exog_list,
                            "early_stop_patience_steps": 15,
                            "val_check_steps": 1
                        },
                        backend=backend
                    ),
                    num_samples=num_samples
            )
        ]

        return NeuralForecast(models=models, freq=freq)

    elif model_type == "multivariate":
        raise Exception("Not implemented yet")
    else:
        raise Exception(f"Unrecognized model type: {model_type}. Available model types are: univariate and multivariate")