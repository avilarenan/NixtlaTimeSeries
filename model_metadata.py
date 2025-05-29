from neuralforecast.models import MLP, LSTM, NHITS, TFT, NBEATSx, TiDE, TSMixerx, BiTCN, DeepNPTS, GRU, TCN
from neuralforecast.losses.pytorch import MAE, MSE, RMSE, MAPE, SMAPE
import pandas as pd
import ast
import json

# NOTE: Intended process
# 1. Configure hyperparameter search space and run with automodels
# 2. Once the automodels have been ran, use the best hyperparameter configuration for the fixed hyperparameters

auto_model_names_to_standard_model_class = {
    "AutoTFT": TFT,
    "AutoTSMixerx": TSMixerx,
    "AutoGRU": GRU,
    "AutoTCN": TCN,
    "AutoTiDE": TiDE,
    "AutoBiTCN": BiTCN,
    "AutoDeepNPTS": DeepNPTS,
    "AutoLSTM": LSTM,
    "AutoNHITS": NHITS,
    "AutoMLP": MLP,
    "AutoNBEATSx": NBEATSx,
}

def get_fixed_hyper_parameter_model(model_name, horizon, hist_exog_list, lookback, loss, dataset_name, use_best_of_trials=False):

    if use_best_of_trials:
        try:
            file_path = f"./results/{dataset_name}/Auto{model_name}_horizon{horizon}_auto/Auto{model_name}_trials.csv"
            print(f"Reading: {file_path}")
            df_trials = pd.read_csv(file_path) # NOTE: reading from already ran trials of auto models
        except Exception as e:
            raise Exception(f"{model_name} does not have the expected auto model (Auto{model_name}) trial result in the following path: {file_path}. \n{e}")
        df_trials = df_trials.sort_values("value")
        df_trials = df_trials.reset_index(drop=True)
        config_str = df_trials.iloc[0].to_dict()["user_attrs_ALL_PARAMS"] # NOTE: selecting the best hyperparameter result

        json_acceptable_string = config_str.replace("'", "\"")
        json_acceptable_string = json_acceptable_string.replace("()", "")
        json_acceptable_string = json_acceptable_string.replace("MSE", "\"MSE\"") # WARNING NOTE: dealing only with MSE, if loss function is changed we should implement it additionally here. 
        json_acceptable_string = json_acceptable_string.replace("False", "\"False\"")
        json_acceptable_string = json_acceptable_string.replace("True", "\"True\"")

        print(f"Model parameters: {json_acceptable_string}")
        config_params_as_dict = json.loads(json_acceptable_string)
        if config_params_as_dict["loss"] == "MSE":
            config_params_as_dict["loss"] = MSE()
            config_params_as_dict["valid_loss"] = MSE()

        for key, value in config_params_as_dict.items():
            if value == "False":
                config_params_as_dict[key] = False
            elif value == "True":
                config_params_as_dict[key] = True

        print(f"Using optimal hyperparameters: [{type(config_params_as_dict)}] {config_params_as_dict}")

        return auto_model_names_to_standard_model_class[f"Auto{model_name}"](**config_params_as_dict)
    else:
        if model_name == "TiDE":
            return TiDE(
                h=horizon,
                input_size=lookback,
                loss=loss,
                layernorm=True,
                num_encoder_layers=2,
                num_decoder_layers=2,
                batch_size=1,
                windows_batch_size=512,
                max_steps=5000,
                val_check_steps=100,
                dropout=0.3,
                learning_rate=0.1,
                early_stop_patience_steps=5,
                temporal_decoder_dim=256,
                decoder_output_dim=8,
                temporal_width=16,
                hist_exog_list=hist_exog_list,
            )
        if model_name == "TFT":
            return TFT(
                h=horizon,
                input_size=lookback,
                loss=loss,
                batch_size=1,
                windows_batch_size=16,
                inference_windows_batch_size=16,
                max_steps=5000,
                val_check_steps=100,
                dropout=0.3,
                learning_rate=0.0198130813398428,
                early_stop_patience_steps=5,
                hist_exog_list=hist_exog_list,
                n_head=4,
                hidden_size=64
            )
        if model_name == "LSTM":
            return LSTM(
                h=horizon, 
                input_size=horizon,
                loss=loss,
                encoder_n_layers=2,
                encoder_hidden_size=128,
                decoder_hidden_size=128,
                decoder_layers=2,
                max_steps=1000,
                learning_rate=0.01,
                early_stop_patience_steps=5,
                hist_exog_list=hist_exog_list,
                h_train=1,
            )
        else:
            raise Exception(f"Model {model_name} not implemented.")


def config_model_generator(horizon, model_class, n_series=None, additional_options={}, backend="optuna"):
    def config(trial):
        if n_series is not None:
            config = {**model_class.get_default_config(h=horizon, n_series=n_series, backend=backend)(trial)}
        else:
            config = {**model_class.get_default_config(h=horizon, backend=backend)(trial)}

        config.update(additional_options)

        return config
    return config

def general_config(horizon, input_size, exog_list, model_name):

    def ret(trial):

        def config_generic(trial):
            return {
                "input_size": input_size,
                "hist_exog_list": exog_list,
                "val_check_steps": 20,
                "max_steps": 1000,
                "batch_size": 1,
                "windows_batch_size": trial.suggest_categorical(
                    "windows_batch_size",
                    [16, 64, 256]
                ),
                "inference_windows_batch_size": 16,
                "scaler_type": "standard",
                "learning_rate": trial.suggest_float(             
                    "learning_rate",
                    low=1e-4,
                    high=1e-1,
                    log=True,
                ),
                "random_seed": 1,
                "early_stop_patience_steps": 5,
            }
        
        if model_name == "AutoLSTM":
            print("Using config for AutoLSTM!")
            config = {**config_generic(trial)}
            config.update({
                "encoder_hidden_size" : trial.suggest_categorical(
                    "encoder_hidden_size",
                    [16, 32, 64, 128, 256, 512]
                ),
                "encoder_n_layers" : trial.suggest_int(
                    "encoder_n_layers",
                    low=1,
                    high=8
                ),
                "encoder_bias" : trial.suggest_categorical(
                    "encoder_bias",
                    [True, False]
                ),
                "encoder_dropout" : trial.suggest_categorical(
                    "encoder_dropout",
                    [0.0, 0.1, 0.2, 0.3]
                ),
                "decoder_hidden_size": trial.suggest_categorical(
                    "decoder_hidden_size",
                    [16, 32, 64, 128, 256, 512]
                )
            })
            return config
        elif model_name == "AutoMLP":
            print("Using config for AutoMLP!")
            config = {**config_generic(trial)}
            config.update({
                "hidden_size" : trial.suggest_categorical(
                    "hidden_size",
                    [64, 128, 256, 512, 1024]
                ),
                "num_layers" : trial.suggest_int(
                    "num_layers",
                    low=2,
                    high=8
                ),
            })
            return config
        elif model_name == "AutoNHITS":
            print("Using config for AutoNHITS!")
            config = {**config_generic(trial)}
            config.update({
                "n_freq_downsample" : trial.suggest_categorical(
                    "n_freq_downsample",
                    [
                        [168, 24, 1],
                        [24, 12, 1],
                        [180, 60, 1],
                        [60, 8, 1],
                        [40, 20, 1],
                        [4, 2, 1],
                        [1, 1, 1],
                    ]
                ),
                "n_pool_kernel_size" : trial.suggest_categorical(
                    "n_pool_kernel_size",
                    [
                        [2, 2, 1],
                        [1, 1, 1],
                        [2, 2, 2],
                        [4, 4, 4],
                        [8, 4, 1],
                        [16, 8, 1]
                    ]
                ),
            })
            return config
        elif model_name == "AutoTFT":
            print("Using config for AutoTFT!")
            config = {**config_generic(trial)}
            config.update({
                "hidden_size" : trial.suggest_categorical(
                    "hidden_size",
                    [64, 128, 256]
                ),
                "n_head" : trial.suggest_categorical(
                    "n_head",
                    [4, 8]
                ),
            })
            return config
        elif model_name == "AutoNBEATSx":
            print("Using config for AutoNBEATSx!")
            config = {**config_generic(trial)}
            return config
        elif model_name == "AutoTiDE":
            print("Using config for AutoTiDE!")
            config = {**config_generic(trial)}
            config.update({
                "hidden_size" : trial.suggest_categorical(
                    "hidden_size",
                    [256, 512, 1024]
                ),
                "decoder_output_dim" : trial.suggest_categorical(
                    "decoder_output_dim",
                    [8, 16, 32]
                ),
                "temporal_decoder_dim" : trial.suggest_categorical(
                    "temporal_decoder_dim",
                    [32, 64, 128, 256, 512]
                ),
                "num_encoder_layers" : trial.suggest_categorical(
                    "num_encoder_layers",
                    [1, 2, 3]
                ),
                "num_decoder_layers" : trial.suggest_categorical(
                    "num_decoder_layers",
                    [1, 2, 3]
                ),
                "temporal_width" : trial.suggest_categorical(
                    "temporal_width",
                    [4, 8, 16]
                ),
                "layernorm" : trial.suggest_categorical(
                    "layernorm",
                    [True, False]
                ),
            })
            return config
        elif model_name == "AutoBiTCN":
            print("Using config for AutoBiTCN!")
            config = {**config_generic(trial)}
            config.update({
                "hidden_size" : trial.suggest_categorical(
                    "hidden_size",
                    [16, 32]
                ),
            })
            return config
        elif model_name == "AutoDeepNPTS":
            print("Using config for AutoDeepNPTS!")
            config = {**config_generic(trial)}
            config.update({
                "hidden_size" : trial.suggest_categorical(
                    "hidden_size",
                    [16, 32, 64]
                ),
            })
            return config
        elif model_name == "AutoTSMixerx":
            print("Using config for AutoTSMixerx!")
            config = {**config_generic(trial)}
            config.update({
                "n_series": 1, # NOTE: Although the model is multivariate, here we want to predict only one target series (forecasting with exogenous)
                "n_block": trial.suggest_categorical(
                    "n_block",
                    [1, 2, 4, 6, 8]
                ),
                "ff_dim": trial.suggest_categorical(
                    "ff_dim",
                    [32, 64, 128, 256, 512]
                ),
            })
            return config
        elif model_name == "AutoGRU":
            print("Using config for AutoGRU!")
            config = {**config_generic(trial)}
            config.update({
                "encoder_hidden_size": trial.suggest_categorical(
                    "encoder_hidden_size",
                    [16, 32, 64, 128, 256, 512]
                ),
                "encoder_n_layers":  trial.suggest_int(
                    "encoder_n_layers",
                    low=1,
                    high=8
                ),
                "decoder_hidden_size": trial.suggest_categorical(
                    "decoder_hidden_size",
                    [16, 32, 64, 128, 256, 512]
                ),
            })
            return config
        elif model_name == "AutoTCN":
            print("Using config for AutoTCN!")
            config = {**config_generic(trial)}
            config.update({
                "encoder_hidden_size": trial.suggest_categorical(
                    "encoder_hidden_size",
                    [16, 32, 64, 128, 256, 512]
                ),
                "decoder_hidden_size": trial.suggest_categorical(
                    "decoder_hidden_size",
                    [16, 32, 64, 128, 256, 512]
                ),
            })
            return config
            
        else:
            print("Using default config!")
            return config_generic
        
    return ret