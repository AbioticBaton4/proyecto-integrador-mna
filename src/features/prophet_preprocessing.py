import pandas as pd
import numpy as np
from prophet.make_holidays import make_holidays_df
from ..utils.helpers import get_min_max_years, get_years_list

EPIDEMIC_DATA = [
    {'holiday': 'importación_de_casos', 'ds': '2020-02-28', 'lower_window': 0, 'ds_upper': '2020-03-23'},
    {'holiday': 'transmisión_comunitaria', 'ds': '2020-03-24', 'lower_window': 0, 'ds_upper': '2020-04-20'},
    {'holiday': 'etapa_epidemiológica', 'ds': '2020-04-21', 'lower_window': 0, 'ds_upper': '2022-05-01'}
]

RECOVERY_DATA = [
    {'holiday': 'etapa1', 'ds': '2020-05-14', 'lower_window': 0, 'ds_upper': '2020-05-17'},
    {'holiday': 'etapa2', 'ds': '2020-05-18', 'lower_window': 0, 'ds_upper': '2020-05-31'},
    {'holiday': 'etapa3', 'ds': '2020-06-01', 'lower_window': 0, 'ds_upper': '2022-05-01'}
]

# Mapea un nombre (string) a la lista de datos (variable)
PREDEFINED_EVENT_SETS = {
    "epidemic": EPIDEMIC_DATA,
    "recovery": RECOVERY_DATA
}

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

def get_upper_window_date(data, date_col='ds', upper_col='ds_upper'):
    df = data.copy()
    for ts_col in [date_col, upper_col]:
        df[ts_col] = pd.to_datetime(df[ts_col])
    df['upper_window'] = (df[upper_col] - df[date_col]).dt.days
    return df


def prepare_custom_holidays(event_data, date_col='ds', upper_col='ds_upper'):
    """
    Prepara un DataFrame de festivos.
    
    event_data (str o list): 
        - Si es un 'str', busca en PREDEFINED_EVENT_SETS.
        - Si es una 'list', la usa directamente.
    """
    
    event_list = None
    
    # --- Lógica para obtener la lista de datos ---
    if isinstance(event_data, str):
        # Es un string. Buscar en el registro.
        event_list = PREDEFINED_EVENT_SETS.get(event_data)
        if event_list is None:
            # Si no se encuentra, lanzar un error
            raise ValueError(f"El set de eventos '{event_data}' no se encuentra. Opciones disponibles: {list(PREDEFINED_EVENT_SETS.keys())}")
            
    elif isinstance(event_data, list):
        # Es una lista. Usarla directamente.
        event_list = event_data
        
    else:
        # No es ni string ni lista
        raise TypeError("El argumento 'event_data' debe ser un 'str' (nombre del set) o una 'list' (de diccionarios).")
    
    # 1. Crear el DataFrame
    df = pd.DataFrame(event_list)
    
    # 2. Procesar el DataFrame
    processed_df = get_upper_window_date(df, date_col=date_col, upper_col=upper_col)
    
    # 3. Retornar el resultado
    return processed_df