import logging
import os
import pickle
import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin
from src.model_dev import LightGBMModel

@step
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> RegressorMixin:
    """
    Trains a LightGBM model, saves it to disk, and returns it.
    """
    try:
        os.makedirs("saved_model", exist_ok=True)

        model = LightGBMModel()
        trained_model = model.train(X_train, y_train)

        with open("saved_model/LightGBM.pkl", "wb") as f:
            pickle.dump(trained_model, f)

        return trained_model

    except Exception as e:
        logging.error(f"Error training LightGBM model: {e}")
        raise e
