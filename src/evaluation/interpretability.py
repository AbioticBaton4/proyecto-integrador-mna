# src/evaluation/interpretability.py
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate


def show_importance_summary(importance_df, format='text', n_top=10):
    print(f"Importancia de las features (top {n_top}):\n")
    if format == 'text':
        print(importance_df.head(n_top).to_string(index=False))
    else:
        print(tabulate(importance_df, headers='keys', tablefmt="fancy_grid"))

def get_xbg_feature_importance(model, show_importances=False, n_top=10, format='text'):
    importance = model.get_score(importance_type='gain')
    imp_df = pd.DataFrame({'feature': list(importance.keys()),
                        'gain': list(importance.values())}) \
            .sort_values('gain', ascending=False)
    if show_importances:
        show_importance_summary(imp_df, format=format, n_top=n_top)
    return imp_df


def plot_feature_importance(importance_df):
    """
    Grafica la importancia de las features desde un DataFrame.
    """
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel("Importancia")
    plt.ylabel("Feature")
    plt.title("Importancia de Features")
    plt.gca().invert_yaxis() # La más importante arriba
    plt.tight_layout()
    plt.show()

def show_test_metrics_table(models_metrics, round_digits=4, table_format='github'):
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