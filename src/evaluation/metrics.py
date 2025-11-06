import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from tabulate import tabulate
import pandas as pd


def show_metric_summary(metrics, format='text',mode = 'Entrenamiento'):
    
    if format == 'text':
        print(f"{mode}\t->\tMAE: {metrics['MAE']:.4f}\tRMSE: {metrics['RMSE']:.4f}\tMAPE: {metrics['MAPE']:.4f}")
    else:
        print(f"\nEvaluación del modelo en el conjunto {mode}:")
        print(tabulate(metrics.items(), headers=["Métrica", "Valor"], tablefmt="fancy_grid"))


def evaluate_model(y_true, y_pred, show_metrics=False, mode = 'Train', metric_format='text'):
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred)
    }

    if show_metrics:
        # print(f"\nEvaluación del modelo en el conjunto {mode}:")
        # print(tabulate(metrics.items(), headers=["Métrica", "Valor"], tablefmt="fancy_grid"))
        show_metric_summary(metrics, format=metric_format, mode=mode)
    return metrics

def evaluate_model_performance(train,test,fittedvalues,forecast, label= 'valor', metric_format='tabulate'):
    print("Evaluating model performance...")
    if label:
        # Caso 1: Se proporciona un label, asumimos que son DataFrames/diccionarios
        y_true_train = train[label]
        y_pred_train = fittedvalues[label]
        y_true_test = test[label]
        y_pred_test = forecast[label]
    else:
        # Caso 2: No se proporciona label, asumimos que son arrays/Series
        y_true_train = train
        y_pred_train = fittedvalues
        y_true_test = test
        y_pred_test = forecast
        
    # Ahora el resto del código funciona igual para ambos casos
    train_metrics = evaluate_model(y_true_train, y_pred_train, show_metrics=True, mode='Train', metric_format=metric_format)
    test_metrics = evaluate_model(y_true_test, y_pred_test, show_metrics=True, mode='Test', metric_format=metric_format)
    return {
        "train": train_metrics,
        "test": test_metrics
    }

def show_metrics_table(models_metrics, round_digits=4, table_format='github'):
    """
    Muestra las métricas del conjunto 'test' en un formato tabular, 
    manejando claves en mayúsculas o minúsculas (ej. 'test', 'Test', 'TEST').

    Parámetros:
    - models_metrics (dict): El diccionario anidado con todas las métricas.
    """
    
    test_metrics_dict = {}
    
    # 1. Iteramos sobre cada modelo
    for model_name, datasets in models_metrics.items():
        found_metrics = None
        
        # 2. Buscamos la clave del dataset que coincida con 'test' (ignorando may/min)
        for dataset_key, metrics_values in datasets.items():
            if dataset_key.lower() == 'test':
                found_metrics = metrics_values
                break # Encontramos 'test' para este modelo, salimos del bucle
        
        # 3. Si encontramos métricas de 'test', las añadimos al diccionario
        if found_metrics is not None:
            test_metrics_dict[model_name] = found_metrics
        else:
            # Opcional: Avisar si un modelo no tenía métricas de 'test'
            print(f"Advertencia: No se encontraron métricas 'test' para el modelo '{model_name}'.")

    # 4. Convertimos el diccionario filtrado en un DataFrame
    if not test_metrics_dict:
        print("No se encontraron métricas de 'test' en ningún modelo.")
        return

    df_test_metrics = pd.DataFrame.from_dict(test_metrics_dict, orient='index')
    
    # 5. Asignamos un nombre al índice
    df_test_metrics.index.name = 'Modelo'
    df_test_metrics = df_test_metrics.round(round_digits).sort_values(by='MAPE', ascending=False)
    
    # 6. Mostramos la tabla formateada
    print("--- Métricas de los Modelos (Test Set) ---")
    if table_format:
        print(df_test_metrics.to_markdown(tablefmt=table_format))
    else:
        for mode, metrics in df_test_metrics.iterrows():
            print(f"\n--- Métricas del modelo: {mode} ---")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value}")
