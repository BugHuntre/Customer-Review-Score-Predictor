import os
import pickle
from typing import Any, Type, Union

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer

DEFAULT_FILENAME = "CustomerSatisfactionEnvironment.pkl"


class CustomerSatisfactionMaterializer(BaseMaterializer):
    """Custom materializer for handling model and data serialization."""

    ASSOCIATED_TYPES = (
        str,
        np.ndarray,
        pd.Series,
        pd.DataFrame,
        CatBoostRegressor,
        RandomForestRegressor,
        LGBMRegressor,
        XGBRegressor,
    )

    def handle_input(
        self, data_type: Type[Any]
    ) -> Union[
        str,
        np.ndarray,
        pd.Series,
        pd.DataFrame,
        CatBoostRegressor,
        RandomForestRegressor,
        LGBMRegressor,
        XGBRegressor,
    ]:
        """Loads an object from the artifact URI using pickle."""
        super().handle_input(data_type)
        filepath = os.path.join(self.artifact.uri, DEFAULT_FILENAME)

        with fileio.open(filepath, "rb") as file:
            return pickle.load(file)

    def handle_return(
        self,
        obj: Union[
            str,
            np.ndarray,
            pd.Series,
            pd.DataFrame,
            CatBoostRegressor,
            RandomForestRegressor,
            LGBMRegressor,
            XGBRegressor,
        ],
    ) -> None:
        """Serializes the object and saves it to the artifact URI."""
        super().handle_return(obj)
        filepath = os.path.join(self.artifact.uri, DEFAULT_FILENAME)

        with fileio.open(filepath, "wb") as file:
            pickle.dump(obj, file)
