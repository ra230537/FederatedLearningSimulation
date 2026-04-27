"""Centralização da função de suavização EMA.

Evita duplicação entre utils/plot_accuracy.py, experiments/plot_ablation.py
e experiments/plot_comparison.py.
"""

import numpy as np


def exponential_moving_average(values, alpha=0.1):
    """Suavização por Média Móvel Exponencial (EMA).

    Aplica a fórmula recorrente:
        s_0 = x_0
        s_t = alpha * x_t + (1 - alpha) * s_{t-1}

    Parâmetros:
        values : array-like de valores a suavizar.
        alpha  : fator de suavização (0 < alpha <= 1).
                 Menor → curva mais suave.
                 Maior → mais fiel aos dados originais.

    Retorna np.ndarray do mesmo tamanho que `values`.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return values
    result = np.empty(n)
    result[0] = values[0]
    for i in range(1, n):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result
