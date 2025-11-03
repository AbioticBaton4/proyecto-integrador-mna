import pandas as pd
import numpy as np
from prophet.make_holidays import make_holidays_df
from ..utils.helpers import get_min_max_years, get_years_list

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

def make_weekly_holidays_df(years_range, country='MX'):
    """
    Crea un DataFrame de festivos (MX) ajustado para datos semanales (ISO Lunes).
    
    Aplica una ventana de -6 días (lower) y 0 días (upper) para
    que el efecto de cualquier festivo de la semana (Lun-Dom)
    se asigne al Lunes de esa semana.
    
    Args:
        years_range (list): Lista de años (ej. [2020, 2021, 2022]).
        country (str): Código del país para los festivos (default 'MX').
        
    Returns:
        pd.DataFrame: DataFrame de festivos listo para Prophet.
    """
    holidays = make_holidays_df(years_range, country=country).sort_values('ds').reset_index(drop=True)
    
    # La lógica clave:
    holidays['lower_window'] = -6  # El efecto aplica hasta 6 días ANTES (Lunes)
    holidays['upper_window'] = 0   # El efecto termina el día del festivo
    
    return holidays

def get_weekly_holidays_df(data: pd.DataFrame, date_col='ds', country='MX', forecast_horizon=1):
    """
    Obtiene un DataFrame de festivos semanales basado en los datos.

    Args:
        data (pd.DataFrame): DataFrame con columna de fechas.
        date_col (str): Nombre de la columna de fechas (default 'ds').
        country (str): Código del país para los festivos (default 'MX').
        forecast_horizon (int): Años adicionales para incluir en festivos futuros.

    Returns:
        pd.DataFrame: DataFrame de festivos listo para Prophet.
    """
    start_year, end_year = get_min_max_years(data, date_col=date_col)
    years_range = get_years_list(start_year, end_year, forecast_horizon=forecast_horizon)
    holidays_df = make_weekly_holidays_df(years_range, country=country)
    return holidays_df