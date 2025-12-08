import pandas as pd
# from src.utils.io import print_table
from IPython.display import display

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia los nombres de las columnas de un DataFrame.
    
    Esta función elimina espacios en blanco al inicio y final de los nombres,
    reemplaza espacios intermedios con guiones bajos y convierte todo a minúsculas.

    Args:
        df (pd.DataFrame): DataFrame cuyas columnas se desean limpiar.

    Returns:
        pd.DataFrame: DataFrame con los nombres de las columnas limpiados.
    """

    df.columns = df.columns.str.strip()               # Elimina espacios al inicio o final
    df.columns = df.columns.str.replace(' ', '_')     # Sustituye espacios por guiones bajos
    df.columns = df.columns.str.lower()               # Convierte todo a minúsculas para consistencia
    return df

def basic_preprocessing(df, new_column_names={}, start_day='1', show_info=False):
    df = clean_column_names(df.copy())

    if new_column_names:
        df.rename(columns=new_column_names, inplace=True)

    df['semana'] = df['semana'].str.replace('sem', '').astype('int')
    df['fecha'] = pd.to_datetime(df['año'].astype(str) + df['semana'].astype(str) + start_day, format='%G%V%u')
    df['valor'] = pd.to_numeric(df['valor'].str.replace(r'[^0-9]', '', regex=True), errors='coerce').fillna(0).astype(int)
    df['padecimiento'] = df['padecimiento'].astype(str).str.replace('\n', '')
    df['codigo_padecimiento'] = df['padecimiento'].str.extract(r'REV\.(.*)')

    if show_info:
        print("Información del DataFrame después del preprocesamiento:")
        print(df.info())
        print("\nPrimeras filas del DataFrame:")
        display(df.head())
    return df

def drop_columns_by_name(df, columns_to_delete):
    """
    Elimina columnas específicas de un DataFrame.

    Args:
        df (pd.DataFrame): DataFrame del cual se eliminarán las columnas.
        columns_to_delete (list): Lista de nombres de columnas a eliminar.

    Returns:
        pd.DataFrame: DataFrame con las columnas eliminadas.
    """
    return df.drop(columns=columns_to_delete, errors='ignore')


def drop_columns_by_prefix(df, prefixes):
    """
    Elimina columnas que comienzan con ciertos prefijos.

    Args:
        df (pd.DataFrame): DataFrame del cual se eliminarán las columnas.
        prefixes (list or str): Prefijo(s) de columnas a eliminar.

    Returns:
        pd.DataFrame: DataFrame sin las columnas que comienzan con los prefijos especificados.
    """
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    
    mask = df.columns.str.startswith(tuple(prefixes))
    return df.loc[:, ~mask]


def delete_columns(df, columns_to_delete=None, prefixes_to_delete=None):
    """
    Función de conveniencia que combina eliminación por nombre y por prefijo.
    
    Args:
        df (pd.DataFrame): DataFrame del cual se eliminarán las columnas.
        columns_to_delete (list, optional): Lista de nombres de columnas a eliminar.
        prefixes_to_delete (list, optional): Lista de prefijos de columnas a eliminar.

    Returns:
        pd.DataFrame: DataFrame con las columnas eliminadas.
    """
    df = df.copy()
    
    if columns_to_delete:
        df = drop_columns_by_name(df, columns_to_delete)
    
    if prefixes_to_delete:
        df = drop_columns_by_prefix(df, prefixes_to_delete)
    
    return df

def basic_fill(df,column):
    df[column] = df[column].fillna(0).astype(int)
    return df

def pivot_df(df, index= ['fecha', 'estado'], columns =['ax_003'], values='valor', show = False):
    # Es más seguro usar pivot_table, maneja cualquier duplicado potencial
    df_pivoteado = df.pivot_table(
        index=index, 
        columns=columns, 
        values=values
    )
    # Opcional: limpiar el resultado para un mejor formato
    df_pivoteado = df_pivoteado.reset_index().rename_axis(None, axis=1)
    if show:
        display(df_pivoteado.head())
    return df_pivoteado