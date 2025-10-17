from statsmodels.tsa.stattools import adfuller, kpss
import numpy as np
import pandas as pd
from tabulate import tabulate
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

class StationarityAnalyzer:
    def __init__(self, series: pd.Series, signif=0.05):
        self.series = series
        self.signif = signif
        self.adf_results_ = {}  # Para guardar los resultados del test ADF
        self.kpss_results_ = {} # Para guardar los resultados del test KPSS

    def __get_crit_value(self, test_type = 'ADF'):

        crit_values_dict = self.adf_results_['valores_criticos'] if test_type == 'ADF' else self.kpss_results_['valores_criticos']
        """Método privado para procesar los valores críticos del test."""
        crit_vals_list = [(key, value) for key, value in crit_values_dict.items()]
        # Busca el valor crítico que corresponde al nivel de significancia
        crit_val = next((value for key, value in crit_vals_list if float(key.strip('%')) == self.signif * 100), None)

        # --- CORRECCIÓN AQUÍ ---
        # Define la clave estándar
        key_name = f'crit_val_{self.signif*100}'
        
        # Usa un if/else para la asignación
        if test_type == 'ADF':
            self.adf_results_[key_name] = crit_val
        else:
            self.kpss_results_[key_name] = crit_val

    def __get_analysis_table(self, test_type='ADF'):
        """Método privado para crear la tabla de resumen del análisis."""
        self.__get_crit_value(test_type) # Esto ahora guarda el valor crítico correctamente
        
        statistic = self.adf_results_['estadistico'] if test_type == 'ADF' else self.kpss_results_['estadistico']
        p_value = self.adf_results_['p_value'] if test_type == 'ADF' else self.kpss_results_['p_value']
        
        # Esta línea ahora funcionará porque la clave es la misma para ambos
        key_name = f'crit_val_{self.signif*100}'
        crit_val = self.adf_results_[key_name] if test_type == 'ADF' else self.kpss_results_[key_name]

        analysis_table = [
            ("Métrica", "Valor Prueba", f'Valor Crítico ({self.signif*100}%)'),
            (f"Estadístico {test_type}", f"{statistic:.4f}", f"{crit_val:.4f}"),
            ("p-value", f"{p_value:.4f}", self.signif),
        ]
        
        # --- CORRECCIÓN AQUÍ ---
        # Usa un if/else para la asignación
        if test_type == 'ADF':
            self.adf_results_['analysis_table'] = analysis_table
        else:
            self.kpss_results_['analysis_table'] = analysis_table
    def _adf_test(self):
        """
        Ejecuta la prueba Dickey-Fuller Aumentada (ADF).

        H0: La serie tiene una raíz unitaria (no es estacionaria).

        H1: La serie no tiene una raíz unitaria (es estacionaria).
        """
        statistic, p_value, _, _, crit_values, _ = adfuller(self.series)

        self.adf_results_ = {
            'estadistico': statistic,
            'p_value': p_value,
            'valores_criticos': crit_values
        }

    def _kpss_test(self, regression='c'):
        """
        Ejecuta la prueba Kwiatkowski-Phillips-Schmidt-Shin (KPSS).

        H0: La serie es estacionaria alrededor de una constante (o tendencia).

        H1: La serie tiene una raíz unitaria (no es estacionaria).
        
        Args:
            regression (str): 'c' para estacionariedad alrededor de una constante (default), 
                              'ct' para estacionariedad alrededor de una tendencia.
        """
        statistic, p_value, _, crit_values = kpss(self.series, regression=regression)

        self.kpss_results_ = {
            'estadistico': statistic,
            'p_value': p_value,
            'valores_criticos': crit_values
        }

    def __tabulate_results(self, test_type='ADF'):
        """
        Método privado para tabular y mostrar los resultados del análisis.
        """
        analysis_table = self.adf_results_['analysis_table'] if test_type == 'ADF' else self.kpss_results_['analysis_table']
        print(tabulate(analysis_table, headers='firstrow', tablefmt='fancy_grid'))

    def run_test(self, test_type='ADF', show_report=True, regression='c'):
        """
        Ejecuta la prueba de estacionariedad especificada y guarda los resultados.

        Args:
            test_type (str): Tipo de prueba a ejecutar ('ADF' o 'KPSS').
        """
        if test_type == 'ADF':
            self._adf_test()
            self.__get_analysis_table(test_type=test_type)
        elif test_type == 'KPSS':
            self._kpss_test(regression=regression)
            self.__get_analysis_table(test_type=test_type)
        else:
            raise ValueError("test_type debe ser 'ADF' o 'KPSS'")
        
        if show_report:
            print(f"\nResultados de la prueba {test_type}:")
            self.__tabulate_results(test_type=test_type)


def analyze_decomposition(series, model='additive', plot=True, figsize=(10, 8)):
    """
    Calcula la descomposición de la serie temporal y opcionalmente la grafica.
    """
    decomposition = seasonal_decompose(series, model=model)

    if plot:
        fig = decomposition.plot()
        fig.set_size_inches(*figsize)
        plt.suptitle('Descomposición de la Serie Temporal', fontsize=16)
        plt.tight_layout()
        plt.show()
        
    # Siempre devuelve el objeto de descomposición para análisis posterior
    return decomposition

