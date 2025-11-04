import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from typing import Literal, Optional
from prophet.plot import plot_plotly, plot_components_plotly

def plot_last_year(df_test, y_test, y_pred, title, plotly_engine=False):
    fechas = df_test['fecha']

    xlabel = 'Fecha'
    ylabel = 'Casos semanales (valor)'

    if plotly_engine:
        if isinstance(fechas.iloc[0], np.datetime64):
            fechas = fechas.dt.strftime('%Y-%m-%d')  # Ensure proper formatting for Plotly

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fechas, y=y_test, mode='lines', name='Real', line=dict(color='steelblue', width=2)))
        fig.add_trace(go.Scatter(x=fechas, y=y_pred, mode='lines', name='Predicho', line=dict(color='darkorange', dash='dash', width=2)))

        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            legend_title='Series',
            template='plotly_white',
            width=1000,
            height=400
        )
        fig.show()
    else:
        plt.figure(figsize=(14,5))
        plt.plot(fechas, y_test, label='Real', color='steelblue', linewidth=2)
        plt.plot(fechas, y_pred, '--', label='Predicho', color='darkorange', linewidth=2)
        plt.title(title)
        plt.xlabel(xlabel); plt.ylabel(ylabel)
        plt.grid(alpha=0.2); plt.legend(); plt.tight_layout(); plt.show()


def plot_residuals_distribution(y_true, y_pred, figsize=(12, 6),
                             title = 'Análisis de Errores',
                             plotly_engine = False,
                             label='valor_original'):
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

def plot_prophet_components(model, forecast, plotly_engine=False, title='Prophet Components'):
    if plotly_engine:
        fig = plot_components_plotly(model, forecast)
        fig.update_layout(title=title)
        fig.show()
    else:
        fig = model.plot_components(forecast)
        plt.title(title)
        plt.show()


def plot_prophet_forecast(model, forecast, plotly_engine=False, title='Default Prophet Forecast', split_date=None):
    if plotly_engine:
        fig = plot_plotly(model, forecast)
        if split_date is not None:
            fig.add_vline(
                x=str(split_date),
                line_width=2,
                line_dash="dash",
                line_color="gray"
            )
            fig.add_annotation(
                                x=str(split_date), # La misma fecha de la línea
                                y=.95,                       # Posición vertical (95% hacia arriba)
                                yref="paper",                 # Coordenadas relativas al área del gráfico (no a los datos y)
                                text="Split Train/Test",      # Tu texto
                                showarrow=False,              # No mostrar una flecha
                                font=dict(color="gray"),      # Color de la fuente
                                xanchor="right",               # Anclar el texto por su lado izquierdo
                                xshift=-10                     # Moverlo 10 píxeles a la izquierda de la línea
                                )
        fig.update_layout(title=title, hovermode="x unified")
        fig.show()
    else:
        fig = model.plot(forecast)
        # Ajusta el tamaño de la figura (matplotlib usa figsize en pulgadas)
        if split_date is not None:
            plt.axvline(x=str(split_date), color='gray', linestyle='--', linewidth=2)
        plt.axhline(y=0, color='red')
        plt.title(title)
        plt.show()


def plot_predictions(forecast_df, data, split_date, 
                  title='Forecast con Banda de Confianza', 
                  x_title='ds', y_title='y', 
                  width=None, height=None, split_title='Split Train/Test'):
    """
    Genera un gráfico de forecast interactivo con Plotly.

    Parámetros:
    - forecast_df (pd.DataFrame): DataFrame con las predicciones (debe tener 'ds', 'yhat', 'yhat_upper', 'yhat_lower').
    - actuals_df (pd.DataFrame): DataFrame con los datos reales (debe tener 'ds', 'y').
    - split_date (str o Timestamp): La fecha (como string o timestamp) donde se hizo la división train/test.
    - title (str): Título del gráfico.
    - x_title (str): Título del eje X.
    - y_title (str): Título del eje Y.
    - width (int): Ancho de la figura en píxeles.
    - height (int): Alto de la figura en píxeles.
    
    Retorna:
    - fig (go.Figure): La figura de Plotly.
    """
    
    fig = go.Figure()
    required_cols_set = {'yhat_upper', 'yhat_lower'}

    if required_cols_set.issubset(forecast_df.columns):
    # --- 1. Banda de Confianza ---
        # Límite superior (oculto)
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat_upper'],
            mode='lines',
            line=dict(width=0),
            name='Límite Superior',
            showlegend=False
        ))

        # Límite inferior (con relleno)
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat_lower'],
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(173, 216, 230, 0.5)',
            fill='tonexty',
            name='Límite Inferior', 
            showlegend=False
        ))

    # --- 2. Línea de Predicción ---
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat'],
        mode='lines',
        line=dict(color='rgb(0, 102, 153)', width=2),
        name='Forecast'
    ))

    # --- 3. Datos Reales (Puntos) ---
    fig.add_trace(go.Scatter(
        x=data['ds'],
        y=data['y'],
        name='Datos Reales',
        mode='markers',
        marker=dict(color='black', size=4),
    ))

    # --- 4. Línea de Split Train/Test ---
    # Convertimos a string para evitar errores de tipo
    if split_date is not None:
        split_date_str = str(split_date)

        fig.add_vline(
            x=split_date_str,
            line_width=2,
            line_dash="dash",
            line_color="gray"
        )

        fig.add_annotation(
            x=split_date_str,
            y=0.95,
            yref="paper",
            text="Split Train/Test",
            showarrow=False,
            font=dict(color="gray"),
            xanchor="right",
            xshift=-10
        )

    # --- 5. Layout y Estilo ---
    fig.update_layout(
        title=title,
        yaxis_title=y_title,
        hovermode="x unified",
        legend_title_text='Componentes',
        margin=dict(l=40, r=40, b=40, t=80),
        width=width,  # Controla el ancho
        height=height,  # Controla el alto
        xaxis=dict(
            title=x_title,  
            rangeslider=dict(
                visible=True  # Esto activa la barra deslizadora
            )
        )
    )
    fig.show()
    return fig