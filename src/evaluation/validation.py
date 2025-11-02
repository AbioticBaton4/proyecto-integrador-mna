from prophet.plot import plot_cross_validation_metric
from prophet.diagnostics import cross_validation, performance_metrics
from IPython.display import display
import warnings


warnings.filterwarnings("ignore")


def cross_validation_prophet(model, initial_weeks, period_weeks, horizon_weeks,
                            plot_cv_metric=False, metric='mape',
                            show_performance_metrics=False):

    df_cv = cross_validation(model,
                              initial=f'{initial_weeks} W',
                              period=f'{period_weeks} W',
                              horizon=f'{horizon_weeks} W')
    if show_performance_metrics:
        print(f"\n--- Métricas de Rendimiento Promedio (Horizonte = {horizon_weeks // 52} año) ---")
        df_p = performance_metrics(df_cv)
        display(df_p.tail())
        
    if plot_cv_metric:
        import matplotlib.pyplot as plt
        fig = plot_cross_validation_metric(df_cv, metric=metric, point_color='blue')
        ax = fig.gca()
        ax.set_title(f'Métrica de Validación Cruzada: {metric.upper()}')
        plt.show()



    return df_cv