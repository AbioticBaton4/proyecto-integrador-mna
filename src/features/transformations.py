import numpy as np
import pandas as pd
from scipy import stats

def get_cummulative_data(df,column, group_by):
    df = df.groupby(group_by)[column].sum().reset_index()
    df['valor_acumulado'] = df[column].cumsum()
    return df

def get_original_scale(series, method='log', lambda_val=None):
    """
    Convierte una serie transformada de vuelta a su escala original.
    
    Args:
        series: Serie de pandas con valores transformados
        method: Método de transformación ('log' o 'boxcox')
        lambda_val: Parámetro lambda para Box-Cox (requerido si method='boxcox')
    
    Returns:
        Serie en escala original
    """
    if method == 'log':
        return np.expm1(series)
    elif method == 'boxcox':
        if lambda_val is None:
            raise ValueError("El parámetro lambda_val es requerido para la transformación Box-Cox.")
        return stats.inv_boxcox(series, lambda_val)
    elif method == 'passthrough':
        return series
    else:
        raise ValueError(f"Método '{method}' no soportado. Use 'log' o 'boxcox'.")
    

def get_original_scale_dataframe(data, method='log', lambda_val=None, original_col_name='valor_original', scaled_col_name='valor'):
    """
    Convierte datos transformados a DataFrame con escala original incluida.
    
    Args:
        data: Serie o DataFrame con valores transformados
        method: Método de transformación ('log' o 'boxcox')
        lambda_val: Parámetro lambda para Box-Cox (requerido si method='boxcox')
        original_col_name: Nombre para la columna con valores en escala original
        scaled_col_name: Nombre para la columna con valores transformados
    
    Returns:
        DataFrame con fecha, valores transformados y valores en escala original
    """
    # Validación y conversión de tipo de entrada
    if isinstance(data, pd.DataFrame):
        df_data = data
    elif isinstance(data, pd.Series):
        df_data = data.to_frame()
    else:
        raise ValueError("El argumento 'data' debe ser una Serie o DataFrame de pandas.")
    
    # Crear copia y preparar columnas
    df_copy = df_data.copy().reset_index()
    df_copy.columns = ['fecha', scaled_col_name]
    
    # Convertir a escala original
    if method != 'passthrough':
        df_copy[original_col_name] = get_original_scale(df_copy[scaled_col_name], method=method, lambda_val=lambda_val)
    
    return df_copy