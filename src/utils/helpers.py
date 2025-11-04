import pandas as pd
from tabulate import tabulate
from datetime import datetime, timedelta
import holidays
import plotly.express as px
import seaborn as sns
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from typing import Literal, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error
from plotly.subplots import make_subplots
from IPython.display import display
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
    df = clean_column_names(df)
    print("Datos cargados exitosamente.")
    
    if show_info:
        print(f"Dimensiones del DataFrame: {df.shape}")
        print("\nPrimeras filas del DataFrame:")
        # print(tabulate(df.head(num_rows), headers='keys', tablefmt='rounded_outline', showindex=False))
        display(df.head(num_rows))
        display(df.info())
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

def plot_timeseries(df,x='fecha',y ='valor_acumulado', 
                    title='Serie de tiempo general "Acumulada"', 
                    plotly_engine=False, figsize=(12, 6), 
                    hover_mode : Optional[Literal['x unified', 'y unified']] = None):
    df = df.copy()
    if plotly_engine:
        if 'semana' not in df.columns:
            df['semana'] = df['fecha'].dt.isocalendar().week
        fig = px.line(df, x=x, y=y,title=title,
              hover_data={'semana': True, 'fecha': True},
              markers=True,
            )
        # Ajustar el tamaño de los marcadores
        fig.update_traces(marker=dict(size=4))
        fig.update_layout(
            width=int(figsize[0]*80),
            height=int(figsize[1]*80),
            hovermode=hover_mode)  # hover aparece dentro de la gráfica)
        fig.show()
    else:
        fig, ax = plt.subplots(figsize=figsize)
        sns.lineplot(data=df, x=x, y=y, marker='o', ax=ax)
        ax.set_title(title)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(True)
        plt.show()

def get_cummulative_data(df,column, group_by):
    df = df.groupby(group_by)[column].sum().reset_index()
    df['valor_acumulado'] = df[column].cumsum()
    return df

def show_decomposition(ts, model='additive', return_decomposition=False, figsize=(10, 8)):
    decomposition = seasonal_decompose(ts, model=model)
    fig = decomposition.plot()
    fig.set_size_inches(*figsize)
    plt.suptitle('Descomposición de la Serie Temporal', fontsize=16)
    plt.tight_layout()
    plt.show()
    if return_decomposition:
        return decomposition
    
# from tabulate import tabulate

def get_crit_val(crit_values, signif):
    crit_vals = [(key, value) for key, value in crit_values.items()]
    crit_val = next((value for key, value in crit_vals if float(key.strip('%')) == signif*100), None)
    return crit_vals, crit_val

def get_analysis_table(tipo='ADF', estadistico=None, p_value=None, crit_val=None, signif=None):
    analysis_table = [
        ("Métrica", "Valor Prueba", 'Nivel de Significancia'),
        (f"Estadístico {tipo}", estadistico, crit_val),
        ("p-value", p_value, signif),
    ]
    return analysis_table

def adf_test(series, signif=0.05, show_crit_vals=False, show_analysis_table=True):
    result = adfuller(series)
    print('Resultados de la Prueba ADF')
    estadistico, p_value, used_lag, n_obs, crit_values, icbest = result
    # print('Estadistico ADF:', estadistico)
    # print('p-value:', p_value)
    # Obtener valores criticos y valor critico para el nivel de significancia dado
    crit_vals, crit_val = get_crit_val(crit_values, signif)
    if show_crit_vals:
        print('Valores críticos:')
        print(tabulate(crit_vals, headers=["Nivel", "Valor crítico"], tablefmt="rounded_outline"))
    if show_analysis_table:
        analysis_table = get_analysis_table(tipo='ADF', estadistico=estadistico, p_value=p_value, crit_val=crit_val, signif=signif)
        print(tabulate(analysis_table, headers='firstrow', tablefmt="fancy_grid"))
    return estadistico, p_value, crit_val


def kpss_test(series, signif=0.05, show_crit_vals=False, show_analysis_table=True):
    result = kpss(series, regression='c')
    estadistico, p_value, lags, crit_values = result
    print('\nResultados de la Prueba KPSS')
    # print('Estadistico KPSS:', estadistico)
    # print('p-value:', p_value)
    crit_vals, crit_val = get_crit_val(crit_values, signif)
    if show_crit_vals:
        print('Valores críticos:')
        print(tabulate(crit_vals, headers=["Nivel", "Valor crítico"], tablefmt="rounded_outline"))
    if show_analysis_table:
        analysis_table = get_analysis_table(tipo='KPSS', estadistico=estadistico, p_value=p_value, crit_val=crit_val, signif=signif)
        print(tabulate(analysis_table, headers='firstrow', tablefmt="fancy_grid"))

    return estadistico, p_value, crit_val

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

def plot_forecast(*dataframes, x='fecha', y='valor_log', 
                  title = 'Comparación de Serie Original, Ajustada y Pronosticada', 
                  figsize=(12, 6), plotly_engine=False, mode='lines',
                  hover_mode : Optional[Literal['x unified', 'y unified']] = None):
    if plotly_engine:
        fig = go.Figure()

        for df, label in dataframes:
            df_copy = df.copy()
            if 'semana' not in df.columns:
                df_copy['semana'] = df_copy['fecha'].dt.isocalendar().week
            fig.add_trace(go.Scatter(
                x=df_copy[x], 
                y=df_copy[y], 
                mode=mode, 
                name=label,
                hovertemplate=
                    f'<b>{label}</b><br>' +
                    'Fecha: %{x}<br>' +
                    f'{y}: ' + '%{y}<br>' +
                    'Semana: %{customdata[0]}<extra></extra>',
                customdata=df_copy[['semana']].values
            ))

        fig.update_layout(
            title=title, 
            xaxis_title=x, 
            yaxis_title=y,
            width=int(figsize[0]*80),
            height=int(figsize[1]*80),
            hovermode=hover_mode,  # hover aparece dentro de la gráfica
            legend_title='Series'
        )
        fig.show()
    else:
        fig, ax = plt.subplots(figsize=figsize)

        for df, label in dataframes:
            sns.lineplot(data=df, x=x, y=y, label=label, ax=ax)

        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        plt.show()


def evaluate_model(y_true, y_pred, show_metrics=False, mode = 'Entrenamiento'):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    if show_metrics:
        print(f"Evaluación del modelo en el conjunto de {mode}:")
        metrics_table = [
            ["Métrica", "Valor"],
            ["MAE", mae],
            ["RMSE", rmse]
        ]
        print(tabulate(metrics_table, headers="firstrow", tablefmt="fancy_grid"))
    return mae, rmse



def plot_residuals_over_time(y_true, y_pred, figsize=(12, 6),
                             title = 'Análisis de Errores',
                             plotly_engine = False,
                             label='valor'):
    # Configuraciones comunes
    xlabel_hist = 'Errores'
    ylabel_hist = 'Frecuencia'
    xlabel_scatter = 'Valores Reales'
    ylabel_scatter = 'Predicciones'
    hist_title = 'Distribución de Errores'
    scatter_title = 'Valores Reales vs Predicciones'

    y_true_aligned = y_true[label]
    y_pred_aligned = y_pred[label]
    if not y_true_aligned.index.equals(y_pred_aligned.index):
        # Reindexa y_pred para que coincida con y_true
        y_pred_aligned = y_pred_aligned.set_axis(y_true_aligned.index)
    x = [min(y_true_aligned), max(y_true_aligned)]
    y = [min(y_true_aligned), max(y_true_aligned)]
    residuals = y_true_aligned - y_pred_aligned
    if plotly_engine:
        fig = make_subplots(rows=1, cols=2, subplot_titles=[hist_title, scatter_title])

        fig.add_trace(
            go.Histogram(x=residuals, nbinsx=30, name='Errores', marker_color='blue', opacity=0.7),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=y_true_aligned, y=y_pred_aligned, mode='markers', name='Predicciones', marker=dict(color='orange')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                # y=[min(y_pred_aligned), max(y_pred_aligned)],
                mode='lines',
                line=dict(color='black', dash='dash'),
                showlegend=False
            ),
            row=1, col=2
        )
        fig.update_xaxes(title_text=xlabel_hist, row=1, col=1)
        fig.update_yaxes(title_text=ylabel_hist, row=1, col=1)
        fig.update_xaxes(title_text=xlabel_scatter, row=1, col=2)
        fig.update_yaxes(title_text=ylabel_scatter, row=1, col=2)
        fig.update_layout(width=int(figsize[0]*80),
                          height=int(figsize[1]*80), 
                          title_text=title)
        fig.show()
    else:
        plt.figure(figsize=figsize)
        plt.subplot(1, 2, 1)
        sns.histplot(residuals, bins=30)
        plt.title(hist_title)
        plt.xlabel(xlabel_hist)
        plt.ylabel(ylabel_hist)
        plt.subplot(1, 2, 2)
        plt.scatter(y_true_aligned, y_pred_aligned)
        plt.plot(x, y, 'k--', lw=2)
        plt.title(scatter_title)
        plt.xlabel(xlabel_scatter)
        plt.ylabel(ylabel_scatter)
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

def split_series(ts, train_size=0.8, show_report=True):
    """
    Divide una serie temporal en conjuntos de entrenamiento y prueba.

    Parámetros:
    ts (pd.Series): Serie temporal a dividir.
    train_size (float): Proporción de datos para el conjunto de entrenamiento.

    Retorna:
    pd.Series, pd.Series: Conjuntos de entrenamiento y prueba.
    """
    split_index = int(len(ts) * train_size)
    train = ts.iloc[:split_index]
    test = ts.iloc[split_index:]
    if show_report:
        print(f"Total de datos: {len(ts)}")
        print(f"Tamaño del conjunto de entrenamiento: {len(train)}")
        print(f"Tamaño del conjunto de prueba: {len(test)}")
    return train, test

def evaluate_model_performance(train,test,fittedvalues,forecast, label= 'valor'):
    print("Evaluating model performance...")
    train_mae, train_mse = evaluate_model(train[label], fittedvalues[label], show_metrics=True)
    test_mae, test_mse = evaluate_model(test[label], forecast[label], show_metrics=True)
    
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

def split_ts(ts, test_size=52, show_report=False):
    '''Divide una serie temporal en conjuntos de entrenamiento y prueba.
    Args:
        ts (pd.DataFrame): Serie temporal a dividir.
        test_size (int): Tamaño del conjunto de prueba en número de períodos.
        show_report (bool): Si es True, muestra un informe del tamaño de los conjuntos.
    Returns:
        train (pd.DataFrame): Conjunto de entrenamiento.
        test (pd.DataFrame): Conjunto de prueba.
    '''
    train = ts.iloc[:-test_size]
    test = ts.iloc[-test_size:]
    if show_report:
        print(f"Total de datos: {len(ts)}")
        print(f"Tamaño del conjunto de entrenamiento: {len(train)}")
        print(f"Tamaño del conjunto de prueba: {len(test)}")
    return train, test