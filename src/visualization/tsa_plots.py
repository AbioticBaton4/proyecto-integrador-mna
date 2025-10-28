import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Literal, Optional
from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss

from statsmodels.graphics.tsaplots import (
    plot_acf as sm_plot_acf, 
    plot_pacf as sm_plot_pacf
)


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


def plot_acf_series(ts, lags=52, use_plotly=False, title='Función de Autocorrelación (ACF)'):
    """
    Calcula y muestra la ACF de una serie temporal.
    Devuelve (acf_values, confint).

    """
    # Preparar serie
    series = ts.dropna()
    if series.empty:
        raise ValueError("La serie está vacía.")
    
    # Calcular ACF y límites de confianza (alpha=0.05)
    acf_vals, confint = acf(series, nlags=lags, alpha=0.05, fft=True)
    lags_arr = np.arange(len(acf_vals))

    if use_plotly:
        # mitad del ancho del intervalo de confianza para barras de error
        err = (confint[:, 1] - confint[:, 0]) / 2
        fig = go.Figure()
        fig.add_trace(go.Bar(x=lags_arr, y=acf_vals, marker_color='steelblue', name='ACF',
                             error_y=dict(type='data', array=err, visible=True)))
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        fig.update_layout(title=title, xaxis_title='Lag', yaxis_title='ACF', template='plotly_white')
        fig.show()
    else:
        plt.figure(figsize=(10, 4))
        sm_plot_acf(series, lags=lags, alpha=0.05)
        plt.title(title)
        plt.tight_layout()
        plt.show()

    return {
        'Autocorrelacion': acf_vals,
        'Intervalos_Confianza': confint
    }

def plot_pacf_series(ts, lags=52, use_plotly=False, title='Función de Autocorrelación Parcial (PACF)'):
    """
    Calcula y muestra la PACF de una serie temporal.
    Devuelve (pacf_values, confint).
    """
    # 1. Preparar serie (eliminar NaNs si los hay por la diferenciación)
    series = ts.dropna()
    if series.empty:
        raise ValueError("La serie está vacía después de eliminar NaNs.")
    
    # 2. Calcular PACF y límites de confianza (alpha=0.05)
    # método 'ywm' es la opción default y suele ser adecuado
    pacf_vals, confint = pacf(series, nlags=lags, alpha=0.05, method='ywm') 
    lags_arr = np.arange(len(pacf_vals))

    if use_plotly:
        # mitad del ancho del intervalo de confianza para barras de error
        # El confint de pacf ya da los límites, por lo que calculamos la mitad del ancho.
        err = (confint[:, 1] - confint[:, 0]) / 2
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=lags_arr, y=pacf_vals, marker_color='darkred', name='PACF',
                             # Establece las barras de error
                             error_y=dict(type='data', array=err, visible=True)))
        
        # Línea horizontal en 0 (cero)
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        
        # Añadir las líneas del intervalo de confianza
        # Usamos la primera y segunda columna de confint para trazar los límites
        # Línea de confianza superior (ejemplo de cómo se podrían añadir, aunque el error_y ya las cubre)
        fig.add_hline(y=confint[0][1], line_dash="dot", line_color="red", opacity=0.5) 
        # Línea de confianza inferior
        fig.add_hline(y=confint[0][0], line_dash="dot", line_color="red", opacity=0.5)

        fig.update_layout(title=title, xaxis_title='Lag', yaxis_title='PACF', template='plotly_white')
        fig.show()
    else:
        # Opción alternativa con Matplotlib (si no se usa Plotly)
        # import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 4))
        sm_plot_pacf(series, lags=lags, alpha=0.05, method='ywm')
        plt.title(title)
        plt.tight_layout()
        plt.show()

    return {
        'Autocorrelacion_Parcial': pacf_vals,
        'Intervalos_Confianza': confint
    }
