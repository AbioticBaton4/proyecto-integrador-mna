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