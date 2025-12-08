import pandas as pd

def extract_date_limits(data: pd.DataFrame, date_col: str = 'ds', verbose: bool = False) -> tuple:
    """
    Identifica la fecha mínima y máxima dentro del DataFrame.
    """
    # Robustez: Aseguramos que sea datetime para evitar errores al llamar .date()
    # Si ya estás seguro que viene limpio, puedes omitir el pd.to_datetime
    fechas = pd.to_datetime(data[date_col])
    
    min_date = fechas.min()
    max_date = fechas.max()
    
    if verbose:
        print(f"Límites temporales encontrados: {min_date.date()} al {max_date.date()}")
        
    return min_date, max_date


def generate_time_index(start_date, end_date, freq: str = 'W-MON', verbose: bool = False) -> pd.DatetimeIndex:
    """
    Genera el índice temporal completo (calendario) basado en los límites.
    Args:
        start_date (pd.Timestamp): Fecha de inicio.
        end_date (pd.Timestamp): Fecha de fin.
        freq (str): Frecuencia para el índice (default 'W-MON').
        verbose (bool): Si es True, muestra información adicional.
    """
    full_idx = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    if verbose:
        print(f"Índice generado: {len(full_idx)} periodos (Freq: {freq}) desde {start_date.date()} hasta {end_date.date()}")
        
    return full_idx


def extract_unique_combinations(df: pd.DataFrame, columns: str | list[str], verbose: bool = False) -> pd.DataFrame:
    """
    Extrae las combinaciones únicas de las columnas de agrupación (Keys).
    Args:
        df (pd.DataFrame): DataFrame de entrada.
        columns (str | list): Columna(s) para extraer combinaciones únicas.
        verbose (bool): Si es True, muestra información adicional.
    """
    if isinstance(columns, str):
        columns = [columns]
    
    # reset_index es vital para que el índice vuelva a ser 0, 1, 2...
    unique_df = df[columns].drop_duplicates().reset_index(drop=True)
    
    if verbose:
        print(f"Combinaciones únicas encontradas: {len(unique_df)}")
        
    return unique_df

def build_grid(unique_groups: pd.DataFrame, 
               date_range: pd.DatetimeIndex, 
               date_col: str = 'ds') -> pd.DataFrame:
    """
    Genera una 'rejilla' maestra mediante el producto cartesiano de 
    grupos y fechas.
    
    Args:
        unique_groups (pd.DataFrame): DataFrame con las combinaciones únicas (keys).
        date_range (pd.DatetimeIndex): El rango de fechas completo.
        date_col (str): Nombre para la columna de fechas resultante.
    
    Returns:
        pd.DataFrame: DataFrame 'esqueleto' listo para ser llenado.
    """
    
    # Usamos nombres claros internamente
    grid = unique_groups.merge(
        pd.Series(date_range, name=date_col), 
        how='cross'
    )
    
    return grid

def join_data_to_grid(grid: pd.DataFrame, 
                      data: pd.DataFrame, 
                      date_col: str = 'ds', 
                      group_cols: str | list[str] = 'estado') -> pd.DataFrame:
    """
    Proyecta los datos observados sobre la rejilla maestra (grid) para identificar huecos.
    
    Args:
        grid (pd.DataFrame): La rejilla completa (combinaciones + fechas).
        data (pd.DataFrame): Los datos originales (observados).
        date_col (str): Nombre de la columna de fecha.
        group_cols (str | list): Columnas que definen los grupos.
    
    Returns:
        pd.DataFrame: DataFrame unido con columna '_merge' para identificar faltantes.
    """
    
    # Normalización de inputs (Buena práctica)
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    
    merged = grid.merge(
        data, 
        on=group_cols + [date_col], 
        how='left',
        indicator=True 
        )
    
    return merged

def expand_timeseries(df:pd.DataFrame, date_col:str='ds', 
                      group_cols: str | list[str] = 'estado', 
                      date_range: pd.DatetimeIndex | None =None,
                      freq='W-MON', verbose=False
                      )-> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Orquesta la expansión de una serie de tiempo para llenar huecos en fechas.
    
    Returns:
        tuple: (data_expanded, data_gaps)
            - data_expanded: DataFrame original + filas rellenadas (limpio).
            - data_gaps: DataFrame que contiene SOLO las filas que faltaban.
    """
    # Normalización de inputs
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    
    # Copiamos para evitar modificar el original
    data = df.copy()

    # Generar rango de fechas si no se proporciona
    if date_range is None:
        if verbose:
            print('Rango de fechas no proporcionado, calculando desde los datos...')
        start_date, end_date = extract_date_limits(data, date_col=date_col, verbose=verbose)
        date_range = generate_time_index(start_date, end_date, freq=freq, verbose=verbose)
    
    # Construcción de la rejilla
    unique_groups = extract_unique_combinations(data, group_cols, verbose=verbose)
    full_grid = build_grid(unique_groups, date_range, date_col=date_col)
    
    # Proyección de datos sobre la rejilla (Merge)
    data_raw_expanded = join_data_to_grid(full_grid, data, date_col=date_col, group_cols=group_cols)
    
    # Extracción de huecos (Gaps)
    data_gaps = data_raw_expanded[data_raw_expanded['_merge'] == 'left_only'].drop(columns='_merge').copy()
    
    # Limpiar DataFrame expandido final
    data_expanded = data_raw_expanded.drop(columns='_merge')
    return data_expanded, data_gaps