import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from typing import Literal, Optional

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