import pandas as pd
from tabulate import tabulate
from datetime import datetime, timedelta
import holidays


def load_data(path, show_info=True, num_rows=5):
    """
    Carga un archivo Parquet en un DataFrame y opcionalmente muestra información básica.

    Args:
        path (str): Ruta del archivo Parquet.
        show_info (bool, optional): Si es True, muestra información del DataFrame. Por defecto es True.
        num_rows (int, optional): Número de filas a mostrar si show_info es True. Por defecto es 5.

    Returns:
        pd.DataFrame: DataFrame cargado desde el archivo Parquet.
    """
    print(f"Cargando datos desde: {path}")
    df = pd.read_parquet(path)
    
    if show_info:
        print(f"Dimensiones del DataFrame: {df.shape}")
        print("\nPrimeras filas del DataFrame:")
        # print(tabulate(df.head(num_rows), headers='keys', tablefmt='rounded_outline', showindex=False))
        display(df.head(num_rows))    
    return df

def clean_column_names(df):
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
        print(tabulate(df.head(), headers='keys', tablefmt='rounded_outline', showindex=False))
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

def get_depression_data(df):
    """
    Filtra el DataFrame para obtener solo los registros relacionados con depresión (código F32).

    Args:
        df (pd.DataFrame): DataFrame original.

    Returns:
        pd.DataFrame: DataFrame filtrado con registros de depresión.
    """
    return df[df['codigo_padecimiento'] == 'F32'].copy()

def get_date_range(df):
    """
    Obtiene el rango de fechas (mínima y máxima) en la columna 'fecha' del DataFrame.

    Args:
        df (pd.DataFrame): DataFrame que contiene la columna 'fecha'.

    Returns:
        tuple: Una tupla con la fecha mínima y máxima.
    """
    min_date = df['fecha'].min()
    max_date = df['fecha'].max()

    all_dates = pd.date_range(min_date, max_date, freq="W-MON")

    return all_dates

def get_unique_combinations(df, columns):
    """
    Versión optimizada que usa groupby y devuelve tanto las combinaciones 
    únicas como los conteos en una sola operación.
    
    Args:
        df (pd.DataFrame): DataFrame del cual obtener las combinaciones.
        columns (list): Lista de nombres de columnas.
    
    Returns:
        tuple: (combinaciones_unicas, combinaciones_con_conteo)
    """
    # Una sola operación groupby para obtener conteos
    combinations_with_counts = df.groupby(columns).size().reset_index(name='count')
    
    # Extraer solo las combinaciones únicas (sin conteos)
    unique_combinations = combinations_with_counts[columns].copy()
    
    print(f"Total de combinaciones únicas: {len(unique_combinations)}")
    
    return unique_combinations, combinations_with_counts


def get_complete_timeseries(combinaciones_unicas, fechas_completas, df_original, columnas_grupo):
    """
    Crea un producto cartesiano entre combinaciones únicas y fechas completas,
    luego hace merge con los datos originales.
    
    Args:
        combinaciones_unicas (pd.DataFrame): DataFrame con combinaciones únicas
        fechas_completas (pd.DatetimeIndex): Rango completo de fechas
        df_original (pd.DataFrame): DataFrame original con los datos
        columnas_grupo (list): Lista de columnas para el grupo ['estado', 'ax_003']
    
    Returns:
        tuple: (df_merge, faltantes) - DataFrame completo y registros faltantes
    """
    # Producto cartesiano (todas las combinaciones × todas las semanas)
    idx = pd.MultiIndex.from_product(
        [combinaciones_unicas.itertuples(index=False, name=None), fechas_completas],
        names=["combo", "fecha"]
    )
    df_completo = pd.DataFrame(index=idx).reset_index()

    # Expandir las columnas del combo
    df_completo[columnas_grupo] = pd.DataFrame(
        df_completo["combo"].tolist(), index=df_completo.index
    )

    df_completo = df_completo.drop(columns="combo")

    # Unir con el DataFrame original
    merge_columns = ["fecha"] + columnas_grupo
    df_merge = df_completo.merge(
        df_original, on=merge_columns, how="left"
    )

    # Filtrar los registros faltantes
    faltantes = df_merge[df_merge.isna().any(axis=1)]
    
    print(f"Total de registros en serie completa: {len(df_merge):,}")
    print(f"Registros con valores faltantes: {len(faltantes):,}")
    print(f"Registros con datos: {len(df_merge) - len(faltantes):,}")
    
    return df_merge, faltantes

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

def basic_fill(df,column):
    df[column] = df[column].fillna(0).astype(int)
    return df

def asignar_temporada(mes):
    if mes in [12, 1, 2]:
        return 'Invierno'
    elif mes in [3, 4, 5]:
        return 'Primavera'
    elif mes in [6, 7, 8]:
        return 'Verano'
    else:
        return 'Otoño'
    
def encontrar_festivos_en_semana(df, fecha_col='fecha', festivos_mx = holidays.Mexico()):
    """
    Revisa si hay días festivos en la semana y año especificados en una fila.
    """
    # %G = Año ISO, %V = Semana ISO, %u = Día de la semana (1=Lunes)
    fecha_inicio = df[fecha_col]
    
    # La semana tiene 7 días
    dias_de_la_semana = [fecha_inicio + timedelta(days=i) for i in range(7)]
    
    # Lista para guardar los festivos encontrados
    festivos_encontrados = []
    
    for dia in dias_de_la_semana:
        # La librería 'holidays' permite revisar si una fecha es festiva
        if dia in festivos_mx:
            festivos_encontrados.append(f"{festivos_mx.get(dia)} ({dia.strftime('%Y-%m-%d')})")
            
    # Si no se encontraron festivos, devolvemos un texto. Si no, la lista.
    if not festivos_encontrados:
        return 0
    else:
        return 1