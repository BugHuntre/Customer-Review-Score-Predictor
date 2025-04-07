import logging 
from zenml import step
import pandas as pd
from sklearn.base import RegressorMixin
from src.evaluation import MSE, RMSE, R2Score
from typing import Tuple
from typing_extensions import Annotated
from zenml.client import Client
import mlflow

# Safe experiment tracker handling
client = Client()
experiment_tracker = client.active_stack.experiment_tracker
experiment_tracker_name = experiment_tracker.name if experiment_tracker else None

@step
def evaluate_model(model: RegressorMixin,
                   X_test: pd.DataFrame,
                   y_test: pd.DataFrame
    ) -> Tuple[
        Annotated[float, "r2_score"],
        Annotated[float, "rmse"],
    ]:
    """
    Evaluate the model on the test data and log metrics to MLflow.

    Args:
        model: Trained regression model
        X_test: Test features
        y_test: True test target values

    Returns:
        r2_score and RMSE as floats
    """
    try:
        prediction = model.predict(X_test)

        mse = MSE().calculate_score(y_test, prediction)
        r2 = R2Score().calculate_score(y_test, prediction)
        rmse = RMSE().calculate_score(y_test, prediction)

        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("rmse", rmse)

        return r2, rmse

    except Exception as e:
        logging.error(f"Error in evaluating model: {e}")
        raise e
