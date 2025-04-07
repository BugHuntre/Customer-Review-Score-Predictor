from lightgbm import LGBMRegressor

class LightGBMModel:
    """LightGBM Regressor implementation."""

    def train(self, x_train, y_train, **kwargs):
        model = LGBMRegressor(**kwargs)
        model.fit(x_train, y_train)
        return model
