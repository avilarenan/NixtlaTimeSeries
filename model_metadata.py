from neuralforecast.losses import MAE, MSE, RMSE, MAPE, SMAPE

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
                "horizon": horizon,
                "input_size": input_size,
                "exog_list": exog_list,
                "loss": MAE(),
                "max_steps": 2000,
                "batch_size": 1,
                "windows_batch_size": 32,
                "scaler_type": "standard",
                # TODO: add dropout?
                "learning_rate": trial.suggest_float(             
                    "learning_rate",
                    low=1e-4,
                    high=1e-1,
                    log=True,
                ),
                "random_seed": trial.suggest_int(
                    "random_seed",
                    low=1,
                    high=20,
                ),
                "early_stop_patience_steps": trial.suggest_int(
                    "early_stop_patience_steps",
                    low=1,
                    high=20
                ),
            }
        
        if model_name == "AutoLSTM":
            print("Using config for AutoLSTM!")
            config = {**config_generic(trial)}
            config.update({
                "encoder_hidden_size" : trial.suggest_categorical(
                    "encoder_hidden_size",
                    [16, 32, 64, 128]
                ),
                "encoder_n_layers" : trial.suggest_int(
                    "encoder_n_layers",
                    low=1,
                    high=4
                ),
                "context_size": trial.suggest_categorical(
                    "context_size",
                    [5, 10, 50]
                ),
                "decoder_hidden_size": trial.suggest_categorical(
                    "decoder_hidden_size",
                    [16, 32, 64, 128]
                )
            })
            return config
        elif model_name == "AutoMLP":
            print("Using config for AutoMLP!")
            config = {**config_generic(trial)}
            config.update({
                "hidden_size" : trial.suggest_categorical(
                    "hidden_size",
                    [256, 512, 1024]
                ),
                "num_layers" : trial.suggest_int(
                    "num_layers",
                    low=2,
                    high=6
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
                    [32, 64, 128]
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
                "layer_norm" : trial.suggest_categorical(
                    "layer_norm",
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
        else:
            print("Using default config!")
            return config_generic
        
    return ret