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
import itertools







def get_depression_data(df):
    """
    Filtra el DataFrame para obtener solo los registros relacionados con depresión (código F32).

    Args:
        df (pd.DataFrame): DataFrame original.

    Returns:
        pd.DataFrame: DataFrame filtrado con registros de depresión.
    """
    return df[df['codigo_padecimiento'] == 'F32'].copy()









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
















