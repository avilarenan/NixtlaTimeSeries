def config_model_generator(horizon, model_class, additional_options={}, backend="optuna"):
    def config(trial):
        config = {**model_class.get_default_config(h=horizon, backend=backend)(trial)}
        config.update(additional_options)
        return config
    return config