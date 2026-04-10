from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Sequence, Tuple, Dict, Any
from tqdm import tqdm
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import seaborn as sns


def _mahalanobis_stats(
    X: np.ndarray,
    regularizacion: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula estadísticos de Mahalanobis para una matriz 2-D.

    Parameters
    ----------
    X : np.ndarray
        Matriz de forma (n_muestras, n_variables).  
        En este caso se esperan dos columnas: consumo del año anterior
        y consumo actual.
    regularizacion : float, optional
        Valor a añadir en la diagonal de la matriz de covarianza cuando
        su determinante sea cero o casi cero, para asegurar que sea
        invertible.  Por defecto 1 × 10⁻⁶.

    Returns
    -------
    d_mh : np.ndarray
        Distancia de Mahalanobis de cada muestra (longitud n_muestras).
    probs : np.ndarray
        Densidad gaussiana multivariante evaluada en cada muestra.
    cov_matrix : np.ndarray
        Matriz de covarianza (2 × 2) empleada en el cálculo.

    Notes
    -----
    * Si la matriz de covarianza no es invertible, se regulariza
      añadiendo `regularizacion` a la diagonal.
    * La densidad se calcula asumiendo distribución normal multivariante
      con los parámetros del grupo.
    """
    mu = X.mean(axis=0)
    cov_matrix = np.cov(X, rowvar=False, ddof=0)

    # Regularizar si es necesario
    if np.linalg.det(cov_matrix) == 0:
        cov_matrix += np.eye(X.shape[1]) * regularizacion

    inv_cov = np.linalg.inv(cov_matrix)

    diffs = X - mu
    mahal_sq = np.einsum("ij,jk,ik->i", diffs, inv_cov, diffs)
    d_mh = np.sqrt(mahal_sq)

    # Constante de normalización (k = 2)
    norm_const = 1.0 / (2 * np.pi * np.sqrt(np.linalg.det(cov_matrix)))
    probs = norm_const * np.exp(-0.5 * mahal_sq)

    return d_mh, probs, cov_matrix


def _procesar_grupo(
    group_df: pd.DataFrame,
    variables_consumo: Sequence[str],
    id_col: str,
    info_grupo: Dict[str, any],
) -> List[Dict[str, any]]:
    """
    Procesa un único grupo socio-geográfico-temporal y devuelve resultados fila a fila.
    """
    X = group_df[variables_consumo].values
    d_mh, probs, _ = _mahalanobis_stats(X)

    resultados_grupo: List[Dict[str, any]] = []
    for i, idx in enumerate(group_df.index):
        resultados_grupo.append(
            dict(
                {
                    id_col: group_df.at[idx, id_col],
                    "consumo_prev": float(X[i, 0]),
                    "consumo_actual": float(X[i, 1]),
                    "probabilidad": float(probs[i]),
                    "fraud_score": float(d_mh[i]),
                },
                **info_grupo,
            )
        )
    return resultados_grupo

def calcular_fraud_scores(
    df: pd.DataFrame,
    group_cols: Sequence[str] | None = None,
    variables_consumo: Sequence[str] | None = None,
    id_col: str = "CLIENTE_ID",
    min_group_size: int = 3,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Calcula puntajes de fraude basados en la distancia de Mahalanobis,
    mostrando una barra de progreso si `show_progress=True`.
    """
    if group_cols is None:
        group_cols = [
            "MUNICIPIO",
            "BARRIO",
            "ESTRATO",
            "D_CLASE_SERVICIO_MES",
            "year_month",
            "TRANSFORMADOR_ID",
        ]
    if variables_consumo is None:
        variables_consumo = ["consumo_prev_year", "consumo"]

    resultados: List[Dict[str, Any]] = []

    # Agrupación
    grouped = df.groupby(group_cols, observed=True)
    iterator = tqdm(grouped, total=len(grouped), desc="Procesando grupos") if show_progress else grouped

    for keys, grp in iterator:
        if len(grp) < min_group_size:
            continue  # grupo demasiado pequeño
        info_grupo = dict(zip(group_cols, keys))
        resultados.extend(
            _procesar_grupo(grp, variables_consumo, id_col, info_grupo)
        )

    return pd.DataFrame(resultados)

# utilitario: dibuja la elipse gaussiana de 2 D
def _plot_gaussian_ellipse(mu, cov, ax, n_std=2, **kwargs):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    ax.add_patch(Ellipse(xy=mu, width=width, height=height,
                         angle=theta, **kwargs))


def plot_consumo_grupo(
    df: pd.DataFrame,
    *,
    min_obs: int = 500,
    n_std: float = 2.0,
    figsize: tuple[int, int] = (7, 7),
    id_colum: str = 'CLIENTE_ID',
    id_interes=None,
    **filters
) -> None:
    """
    Filtra `df` con los pares columna-valor de **filters y grafica
    consumo_prev vs. consumo_actual.

    • Puntos anómalos (Mahalanobis > n_std σ)   → rojo
    • ID(s) de interés (id_interes)             → dorado
      --> NO se etiquetan ni colorean como anómalos/ normales.

    Parameters
    ----------
    id_interes : int | str | Sequence | None
        ID (o lista/iterable de IDs) que se resaltarán en dorado.
    """

    # 1) Filtrado
    group = df.copy()
    for col, val in filters.items():
        group = group[group[col] == val]

    if group.shape[0] < min_obs:
        print(f"Grupo {filters} omitido (<{min_obs} obs)")
        return

    # 2) Medidas base
    X = group[['consumo_prev', 'consumo_actual']].values
    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False, ddof=0)
    if np.linalg.det(cov) == 0:
        cov += np.eye(2) * 1e-6

    inv_cov = np.linalg.inv(cov)
    diffs = X - mu
    mahal_dist = np.sqrt((diffs @ inv_cov * diffs).sum(axis=1))

    es_anomalo = mahal_dist > n_std

    # 2.1) Máscara para id_interes
    if id_interes is None:
        es_interes = np.zeros_like(es_anomalo, dtype=bool)
    else:
        if not isinstance(id_interes, (list, tuple, set, np.ndarray)):
            id_interes = [id_interes]
        es_interes = group[id_colum].isin(id_interes).values

    # prioridad de color: interés > anómalos > normales
    colores = np.where(
        es_interes, 'gold',
        np.where(es_anomalo, 'crimson', 'royalblue')
    )

    # 3) Plot
    sns.set(style='whitegrid', context='talk', font_scale=1.1)
    fig, ax = plt.subplots(figsize=figsize)

    sns.scatterplot(
        x=X[:, 0], y=X[:, 1],
        hue=colores,
        palette={'royalblue': 'royalblue',
                 'crimson':   'crimson',
                 'gold':      'gold'},
        ax=ax, s=100, edgecolor='k', alpha=0.9, legend=False
    )

    # y = x
    ax.plot([X[:, 0].min(), X[:, 0].max()],
            [X[:, 0].min(), X[:, 0].max()],
            linestyle='--', color='gray', linewidth=1.5)

    # elipse
    _plot_gaussian_ellipse(mu, cov, ax, n_std=n_std,
                           edgecolor='forestgreen', fc='none', lw=2.5)

    # 4) Etiquetas
    #    a) anómalos EXCLUYENDO id_interes
    mask_anom_no_int = es_anomalo & ~es_interes
    ids_anom = group.loc[mask_anom_no_int, id_colum].values
    for (x_pt, y_pt), cid in zip(X[mask_anom_no_int], ids_anom):
        ax.text(x_pt, y_pt, str(cid), fontsize=7, color='crimson', weight='bold',
                ha='right', va='bottom',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none',
                          boxstyle='round,pad=0.2'))

    #    b) id_interes
    if es_interes.any():
        ids_int = group.loc[es_interes, id_colum].values
        for (x_pt, y_pt), cid in zip(X[es_interes], ids_int):
            ax.text(x_pt, y_pt, str(cid), fontsize=8, color='darkgoldenrod',
                    weight='bold', ha='left', va='top')

    # 5) Título
    pretty_title = " | ".join(f"{k}: {v}" for k, v in filters.items())
    ax.set(xlabel='Consumo año anterior',
           ylabel='Consumo año actual',
           title=pretty_title)

    # 6) Leyenda
    legend_elems = [
        Line2D([0], [0], marker='o', color='w', label='Normal',
               markerfacecolor='royalblue', markeredgecolor='k', markersize=12),
        Line2D([0], [0], marker='o', color='w',
               label=f'Anómalo (>{n_std}σ)', markerfacecolor='crimson',
               markeredgecolor='k', markersize=12),
        Line2D([0], [0], linestyle='--', color='gray', lw=2, label='y = x'),
        Line2D([0], [0], linestyle='-', color='forestgreen', lw=2,
               label=f'{n_std}σ')
    ]
    if es_interes.any():
        legend_elems.insert(
            1,
            Line2D([0], [0], marker='o', color='w', label='ID de interés',
                   markerfacecolor='gold', markeredgecolor='k', markersize=12)
        )

    # margen y posición (leyenda fuera del eje)
    fig.subplots_adjust(right=0.8)
    ax.legend(handles=legend_elems,
              loc='upper left', bbox_to_anchor=(1, 1),
              borderaxespad=0, fontsize=12, frameon=False)

    sns.despine()
    plt.tight_layout()
    plt.show()