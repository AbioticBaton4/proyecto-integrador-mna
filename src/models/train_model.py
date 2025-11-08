from prophet import Prophet

def train_prophet_model(data, verbose: bool = False, **prophet_params):
    """
    Entrena un modelo Prophet con los datos proporcionados y parámetros opcionales.

    Parámetros:
    - data (pd.DataFrame): DataFrame con columnas 'ds' (fechas) y 'y' (valores).
    - prophet_params: Parámetros adicionales para el modelo Prophet.

    Retorna:
    - model (Prophet): Modelo Prophet entrenado.
    """
    s_name = prophet_params.pop('custom_seasonality_name', None)
    s_period = prophet_params.pop('custom_period', None)
    s_fourier = prophet_params.pop('custom_fourier_order', None)
    s_prior = prophet_params.pop('custom_prior_scale', None)
    model = Prophet(**prophet_params)
    if s_name:
        if verbose:
            print(f"Adding custom seasonality: {s_name} with period {s_period} and fourier order {s_fourier}")
        model.add_seasonality(
            name=s_name,
            period=s_period,
            fourier_order=s_fourier,
            prior_scale=s_prior
        )
    model.fit(data)
    return model