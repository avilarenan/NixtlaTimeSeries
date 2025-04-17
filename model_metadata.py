def config_model_generator(horizon, model_class, n_series=None, additional_options={}, backend="optuna"):
    def config(trial):
        if n_series is not None:
            config = {**model_class.get_default_config(h=horizon, n_series=n_series, backend=backend)(trial)}
        else:
            config = {**model_class.get_default_config(h=horizon, backend=backend)(trial)}
        config.update(additional_options)
        return config
    return config