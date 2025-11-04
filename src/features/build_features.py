import pandas as pd
import numpy as np

def create_basic_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Crea características de tiempo básicas a partir de una columna de fecha.

    Parámetros:
    -----------
    df (pd.DataFrame): 
        DataFrame de entrada.
    date_col (str): 
        Nombre de la columna que contiene las fechas.
        
    Retorna:
    --------
    pd.DataFrame: 
        Un nuevo DataFrame con las siguientes columnas añadidas:
        'año', 'mes', 'semana', 'trimestre', 'semestre',
        'mes_sin', 'mes_cos'.
    """
    df_feat = df.copy()  # Usar un nombre diferente es buena práctica (opcional)
    
    # Asegurar que sea datetime
    df_feat[date_col] = pd.to_datetime(df_feat[date_col])
    
    # Extraer características
    df_feat['año'] = df_feat[date_col].dt.year
    df_feat['mes'] = df_feat[date_col].dt.month
    df_feat['semana'] = df_feat[date_col].dt.isocalendar().week.astype(int)
    df_feat['trimestre'] = df_feat[date_col].dt.quarter
    df_feat['semestre'] = df_feat['mes'].apply(lambda m: 1 if m <= 6 else 2)
    
    # Características cíclicas (excelente para ML)
    df_feat['mes_sin'] = np.sin(2 * np.pi * df_feat['mes'] / 12)
    df_feat['mes_cos'] = np.cos(2 * np.pi * df_feat['mes'] / 12)

    return df_feat

def prepare_data(data: pd.DataFrame, date_col: str, target_col: str, new_target_name: str) -> pd.DataFrame:
    df = data[[date_col, target_col]].copy()
    df[new_target_name] = df[target_col].diff()
    df.loc[0, new_target_name] = df.iloc[0][target_col]
    df.drop(columns=[target_col], inplace=True)
    return df

def add_lags(df, column, lags):
    """
    Agrega columnas de rezagos (lags) a un DataFrame.

    Parámetros:
    - df: DataFrame original.
    - column: Nombre de la columna a la que se le agregarán los rezagos.
    - lags: Lista de enteros que representan los períodos de rezago.

    Retorna:
    - DataFrame con las columnas de rezagos agregadas.
    """
    df_with_lags = df.copy()
    for lag in lags:
        df_with_lags[f'{column}_lag{lag}'] = df_with_lags[column].shift(lag)
    df_with_lags.dropna(inplace=True)  # Eliminar filas con valores NaN generados por los rezagos
    return df_with_lags