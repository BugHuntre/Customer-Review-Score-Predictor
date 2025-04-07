import logging
import pandas as pd
from src.data_cleaning import DataCleaning, DataPreprocessStrategy

def get_data_for_test(return_json: bool = True):
    """Load and preprocess a sample dataset for testing or prediction.

    Args:
        return_json (bool): If True, returns the preprocessed data as JSON (orient="split").
                            If False, returns a pandas DataFrame.

    Returns:
        str or pd.DataFrame: Preprocessed data as JSON or DataFrame.
    """
    try:
        data_path = r"C:\Users\ASUS\Desktop\olist_customer_satisfaction\data\olist_customers_dataset.csv"
        df = pd.read_csv(data_path).sample(n=100)

        preprocess_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(df, preprocess_strategy)
        cleaned_df = data_cleaning.handle_data()
        cleaned_df.drop(columns=["review_score"], inplace=True)

        if return_json:
            return cleaned_df.to_json(orient="split")
        return cleaned_df

    except Exception as e:
        logging.error("Error in get_data_for_test: %s", e)
        raise
