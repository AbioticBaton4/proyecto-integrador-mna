import pandas as pd
import numpy as np

def prophet_df(data: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
    """
    Prepare DataFrame for Prophet model.

    Parameters:
    data (pd.DataFrame): Input DataFrame containing raw data.
    date_col (str): Name of the date column.
    target_col (str): Name of the target variable column.

    Returns:
    pd.DataFrame: DataFrame formatted for Prophet with 'ds' and 'y' columns.
    """
    data_prophet = data[[date_col, target_col]].copy()
    data_prophet.rename(columns={date_col: 'ds', target_col: 'y'}, inplace=True)
    return data_prophet