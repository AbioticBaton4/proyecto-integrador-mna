from prophet import Prophet
import pandas as pd

# --- 1. Función de Predicción del Modelo Prophet ---
# horizon_weeks, puede ser opcional si el DataFrame 'future' existe completo

def prophet_forecast(model: Prophet, future: pd.DataFrame = None, horizon_weeks: int=None,
                     freq: str = 'W-MON') -> pd.DataFrame:
    """
    Realiza predicciones utilizando un modelo Prophet entrenado.

    Parámetros:
    - model (Prophet): Modelo Prophet entrenado.
    - future (pd.DataFrame): DataFrame con las fechas futuras a predecir.
    - horizon_weeks (int): Número de semanas para predecir hacia adelante.

    Retorna:
    - pd.DataFrame: DataFrame con las predicciones.
    """
    if future is None or future.empty:
        print("El DataFrame 'future' no puede estar vacío.")
        print(f"Generando future dataframe para {horizon_weeks} semanas hacia adelante.")
        if horizon_weeks is None:
            raise ValueError("Debe especificar 'horizon_weeks' si 'future' no está completo.")
        future = model.make_future_dataframe(periods=horizon_weeks,  freq=freq)
    forecast = model.predict(future)
    return forecast
