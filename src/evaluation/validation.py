from prophet.plot import plot_cross_validation_metric
from prophet.diagnostics import cross_validation, performance_metrics
from IPython.display import display
from prophet import Prophet
import pandas as pd
from tqdm import tqdm
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

def tune_prophet_params(data_prophet, all_params, cv_args):
    """
    Realiza un ajuste de hiperparámetros para Prophet usando CV.
    
    Maneja estacionalidades personalizadas pasadas dentro del diccionario
    de parámetros (con el prefijo 'custom_').
    
    Argumentos:
        data_prophet (pd.DataFrame): DataFrame listo para Prophet (columnas 'ds', 'y').
        all_params (list[dict]): Lista de diccionarios, cada uno una config. de hiperparámetros.
        cv_args (dict): Argumentos para pasar a la función cross_validation de Prophet 
                        (ej. initial, period, horizon).
                        
    Devuelve:
        best_params (dict): El diccionario de parámetros con el mejor MAPE.
        tuning_results (pd.DataFrame): DataFrame con todos los parámetros probados y su MAPE.
    """
    
    iterator = tqdm(all_params, desc="Ajustando Hiperparámetros", unit="config")
    mapes = []

    for params in iterator:
        
        current_params = params.copy()

        # 1. Extrae parámetros 'custom' para add_seasonality
        s_name = current_params.pop('custom_seasonality_name', None)
        s_period = current_params.pop('custom_period', None)
        s_fourier = current_params.pop('custom_fourier_order', None)
        s_prior = current_params.pop('custom_prior_scale', None)

        # 2. Inicializa Prophet con los parámetros estándar
        m = Prophet(**current_params) 
        
        # 3. Añade la estacionalidad custom si existe
        if s_name:
            m.add_seasonality(
                name=s_name, 
                period=s_period, 
                fourier_order=s_fourier, 
                prior_scale=s_prior
            )
            
        # 4. Ajuste, CV y métricas
        m.fit(data_prophet)
        df_cv = cross_validation(m, **cv_args)
        df_p = performance_metrics(df_cv, rolling_window=1)
        mapes.append(df_p['mape'].values[0])

    # 5. Compila y devuelve resultados
    tuning_results = pd.DataFrame(all_params)
    tuning_results['mape'] = mapes
    tuning_results = tuning_results.sort_values('mape', ascending=True)

    best_params = tuning_results.iloc[0].to_dict()
    # Limpiamos el 'mape' del diccionario de mejores parámetros
    best_params.pop('mape', None) 

    print("Mejores hiperparámetros encontrados:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
        
    return best_params, tuning_results