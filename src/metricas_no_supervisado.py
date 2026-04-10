"""
Métricas de evaluación para modelos no supervisados de detección de anomalías.

Las métricas asumen que el modelo produce un ranking ordenado de mayor a menor
sospecha, y se evalúa qué tan bien ese ranking recupera los clientes etiquetados
como anómalos (ground truth proveniente de inspecciones).

Funciones disponibles:
    precision_at_k   : fracción de anómalos reales en los primeros k resultados
    recall_at_k      : fracción de anómalos reales cubiertos en los primeros k
    average_precision: AP (área bajo la curva precisión-recall discreta)
    curva_pr_at_k    : precision y recall para cada k (para graficar)
    resumen_metricas : tabla con P@k y R@k para múltiples valores de k
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Sequence


# ---------------------------------------------------------------------------
# Funciones base
# ---------------------------------------------------------------------------

def precision_at_k(retrieved: Sequence, relevant: Sequence, k: int) -> float:
    """
    Fracción de ítems relevantes entre los primeros k recuperados.

    Args:
        retrieved : lista ordenada de IDs según el ranking del modelo
        relevant  : conjunto de IDs que son anómalos reales (ground truth)
        k         : número de resultados a considerar

    Returns:
        Valor entre 0 y 1.
    """
    if k <= 0:
        raise ValueError("k debe ser un entero positivo.")
    retrieved_k = retrieved[:k]
    hits = len(set(retrieved_k) & set(relevant))
    return hits / k


def recall_at_k(retrieved: Sequence, relevant: Sequence, k: int) -> float:
    """
    Fracción de anómalos reales cubiertos en los primeros k recuperados.

    Args:
        retrieved : lista ordenada de IDs según el ranking del modelo
        relevant  : conjunto de IDs que son anómalos reales (ground truth)
        k         : número de resultados a considerar

    Returns:
        Valor entre 0 y 1.
    """
    if not relevant:
        return 0.0
    retrieved_k = set(retrieved[:k])
    hits = len(retrieved_k & set(relevant))
    return hits / len(relevant)


def average_precision(retrieved: Sequence, relevant: Sequence) -> float:
    """
    Average Precision (AP): media de la precisión en cada posición
    donde se recupera un ítem relevante.

    Equivale al área bajo la curva precisión-recall discreta.

    Args:
        retrieved : lista ordenada de IDs según el ranking del modelo
        relevant  : conjunto de IDs que son anómalos reales (ground truth)

    Returns:
        AP score entre 0 y 1.
    """
    if not relevant:
        return 0.0

    relevant_set = set(relevant)
    hits = 0
    sum_precisions = 0.0

    for k, item in enumerate(retrieved, start=1):
        if item in relevant_set:
            hits += 1
            sum_precisions += hits / k

    return sum_precisions / len(relevant_set)


# ---------------------------------------------------------------------------
# Funciones de análisis
# ---------------------------------------------------------------------------

def curva_pr_at_k(
    retrieved: Sequence,
    relevant: Sequence,
    paso: int = 1,
) -> pd.DataFrame:
    """
    Calcula precision@k y recall@k para cada k desde `paso` hasta len(retrieved).

    Args:
        retrieved : lista ordenada de IDs según el ranking del modelo
        relevant  : conjunto de IDs que son anómalos reales (ground truth)
        paso      : incremento entre valores de k (default 1)

    Returns:
        DataFrame con columnas [k, precision, recall].
    """
    relevant_set = set(relevant)
    n = len(retrieved)
    ks = range(paso, n + 1, paso)

    rows = []
    hits_acc = 0
    for k in range(1, n + 1):
        if retrieved[k - 1] in relevant_set:
            hits_acc += 1
        if k % paso == 0:
            rows.append({
                "k":         k,
                "precision": hits_acc / k,
                "recall":    hits_acc / len(relevant_set) if relevant_set else 0.0,
            })

    return pd.DataFrame(rows)


def resumen_metricas(
    retrieved: Sequence,
    relevant: Sequence,
    ks: Sequence[int] | None = None,
) -> pd.DataFrame:
    """
    Tabla resumen de P@k y R@k para múltiples valores de k.

    Args:
        retrieved : lista ordenada de IDs según el ranking del modelo
        relevant  : conjunto de IDs que son anómalos reales (ground truth)
        ks        : lista de valores k a evaluar.
                    Por defecto usa [100, 500, 1000, 2000, 5000, len(retrieved)].

    Returns:
        DataFrame con columnas [k, precision_at_k, recall_at_k].
    """
    if ks is None:
        n = len(retrieved)
        ks = [k for k in [50, 100, 500, 1000, 2000, 3000, 4000, 5000] if k <= n] + [n]

    rows = []
    for k in ks:
        rows.append({
            "k":              k,
            "precision_at_k": precision_at_k(retrieved, relevant, k),
            "recall_at_k":    recall_at_k(retrieved, relevant, k),
        })

    df = pd.DataFrame(rows)
    df["ap"] = average_precision(retrieved, relevant)
    return df
