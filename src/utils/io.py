import pandas as pd
from IPython.display import display
from src.data.cleaning import clean_column_names
from tabulate import tabulate
import itertools

def load_data(path, show_info=False, show_data=False, num_rows=5):
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
    df = clean_column_names(df)
    print("Datos cargados exitosamente.")
    
    if show_info:
        get_basic_info(df)
    if show_data:
        display(df.head(num_rows))
    return df

def get_basic_info(df: pd.DataFrame) -> None:
    """
    Muestra información básica de un DataFrame.

    Args:
        df (pd.DataFrame): DataFrame del cual se mostrará la información.

    """
    print(f"Dimensiones del DataFrame:")
    print(f'Total de registros: {df.shape[0]:,}')
    print(f'Total de columnas: {df.shape[1]}\n')      
    display(df.info())

def print_table(df: pd.DataFrame, title: str = "", headers: str | list[str] | dict[str, str] ='keys', tablefmt='rounded_outline') -> None:
    """
    Imprime un DataFrame en formato de tabla usando tabulate.

    Args:
        df (pd.DataFrame): DataFrame a imprimir.
        headers (str or list, optional): Encabezados para la tabla. Por defecto es 'keys'.
        tablefmt (str, optional): Formato de la tabla. Por defecto es 'pretty'.
    """
    print(title)
    print(tabulate(df, headers=headers, tablefmt=tablefmt, showindex=False)) # pyright: ignore[reportArgumentType]

def save_model_params(params, path):
    import json
    with open(path, 'w') as f:
        json.dump(params, f)
    print(f"Parámetros del modelo guardados en: {path}")

def load_model_params(path):
    import json
    with open(path, 'r') as f:
        params = json.load(f)
    print(f"Parámetros del modelo cargados desde: {path}")
    return params

def setup_logging(root_level=None,
                prophet_level=None,
                cmdstanpy_level=None,
                silence_cmdstanpy=True,
                stream_to_stderr=True):
    """
    Configura logging para el notebook, silenciando cmdstanpy y dejando Prophet en INFO (por defecto).
    Los parámetros None respetan la configuración previa (p. ej. la de la celda 4).
    """
    # Usa niveles por defecto si no se pasan
    import logging
    root_level = root_level or logging.WARNING
    prophet_level = prophet_level or logging.INFO
    cmdstanpy_level = cmdstanpy_level or logging.WARNING

    # Root: baja el ruido global
    root = logging.getLogger()
    root.setLevel(root_level)
    for h in root.handlers:
        h.setLevel(root_level)

    # cmdstanpy: silenciar completamente si se solicita
    logger_cmdstanpy = logging.getLogger('cmdstanpy')
    logger_cmdstanpy.setLevel(cmdstanpy_level)
    if silence_cmdstanpy:
        logger_cmdstanpy.handlers.clear()
        logger_cmdstanpy.propagate = False
        if not any(isinstance(h, logging.NullHandler) for h in logger_cmdstanpy.handlers):
            logger_cmdstanpy.addHandler(logging.NullHandler())

    # prophet: dejar solo sus mensajes con un StreamHandler formateado
    logger_prophet = logging.getLogger('prophet')
    logger_prophet.setLevel(prophet_level)
    logger_prophet.propagate = False
    # evita duplicar handlers de stream
    logger_prophet.handlers = [
        h for h in logger_prophet.handlers
        if not isinstance(h, logging.StreamHandler)
    ]
    if stream_to_stderr:
        handler = logging.StreamHandler()
        handler.setLevel(prophet_level)
        handler.setFormatter(logging.Formatter('%(name)s: %(levelname)s - %(message)s'))
        logger_prophet.addHandler(handler)

def get_years_list(start_year, end_year, forecast_horizon=1, verbose=False):
    """
    Crea una lista de años (int) a partir de un año de inicio,
    fin y un horizonte de pronóstico.
    """
    final_end_year = end_year + forecast_horizon
    
    if verbose:
        # --- Arreglo Aquí ---
        print(f"Generando lista de años desde {start_year} hasta {final_end_year}")
        
    # Agregamos +1 para que range() incluya el último año
    return list(range(start_year, final_end_year + 1))

def get_min_max_years(data, date_col='ds', verbose=False):
    """
    Obtiene el año mínimo y máximo de una columna de fechas.
    
    Returns:
        tuple: (start_year, end_year)
    """
    start_year = data[date_col].dt.year.min()
    end_year = data[date_col].dt.year.max()
    
    if verbose:
        # Arreglamos esto:
        print(f"Rango de años en datos: {start_year} - {end_year}")
        
    return start_year, end_year


def generate_param_list(grid):  
    return [dict(zip(grid.keys(), v)) for v in itertools.product(*grid.values())]

def filter_data()