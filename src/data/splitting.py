import pandas as pd

def split_series(ts: pd.Series, train_size=0.8, show_report=True):
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