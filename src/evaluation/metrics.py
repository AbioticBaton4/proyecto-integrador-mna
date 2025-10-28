import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from tabulate import tabulate


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
