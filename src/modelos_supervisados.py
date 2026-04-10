"""
modelos_supervisados.py
-----------------------
Utilidades compartidas para el entrenamiento supervisado de detección de anomalías.

Funciones:
    construir_features(df, incluir_target)  →  extrae features por producto
    plot_roc_curve(clf, Xva, yva, nombre)   →  grafica curva ROC
"""
from tqdm import tqdm 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

NUM_COLS = [
    "consumo_mean", "consumo_std", "consumo_min", "consumo_max", "consumo_last",
    "pct_mean", "pct_std", "trend",
]
CAT_COLS = ["categoria", "subcategoria", "tipo_producto", "localidad", "barrio"]


def construir_features(
    df: pd.DataFrame,
    incluir_target: bool = True,
) -> tuple:
    """
    Agrega la serie de consumo por producto y extrae features estáticas.

    Parameters
    ----------
    df : DataFrame con columnas producto, fecha, consumo, y categóricas.
         Si incluir_target=True debe tener también 'anomalo'.
    incluir_target : bool
        True  → devuelve (X, y, cat_cols) para entrenamiento
        False → devuelve (X, cat_cols) para predicción en batch

    Returns
    -------
    X : DataFrame con features numéricas y categóricas (una fila por producto)
    y : array de etiquetas (solo si incluir_target=True)
    cat_cols : lista de columnas categóricas presentes en X
    """
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values(["producto", "fecha"])

    def _agg_grupo(g):
        pct = g["consumo"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
        row = {
            "consumo_mean": g["consumo"].mean(),
            "consumo_std":  g["consumo"].std(ddof=0),
            "consumo_min":  g["consumo"].min(),
            "consumo_max":  g["consumo"].max(),
            "consumo_last": g["consumo"].iloc[-1],
            "pct_mean":     pct.mean(),
            "pct_std":      pct.std(ddof=0),
            "trend":        np.polyfit(np.arange(len(g)), g["consumo"].values, 1)[0]
                            if len(g) >= 2 else 0,
        }
        for c in CAT_COLS:
            row[c] = g[c].mode().iloc[0] if c in g.columns and not g[c].isna().all() else np.nan
        if incluir_target:
            row["y"] = (
                g["anomalo"].iloc[0]
                if g["anomalo"].nunique() == 1
                else g["anomalo"].mode().iloc[0]
            )
        return pd.Series(row)

    tqdm.pandas(desc="Construyendo features")   
    agg = df.groupby("producto").progress_apply(_agg_grupo).reset_index()

    cat_cols = [c for c in CAT_COLS if c in agg.columns]
    X = agg[NUM_COLS + cat_cols].copy()

    if incluir_target:
        y = agg["y"].astype(int).values
        return X, y, cat_cols

    return X, cat_cols


def plot_roc_curve(clf, Xva, yva: np.ndarray, nombre: str, ax=None, out_path: str = None):
    """
    Grafica la curva ROC de un clasificador entrenado.

    Parameters
    ----------
    clf       : Pipeline sklearn con método predict_proba
    Xva, yva  : datos de validación
    nombre    : etiqueta del modelo para la leyenda
    ax        : Axes de matplotlib (crea uno nuevo si es None)
    out_path  : ruta para guardar la figura (opcional)
    """
    p = clf.predict_proba(Xva)[:, 1]
    fpr, tpr, _ = roc_curve(yva, p)
    roc_auc = auc(fpr, tpr)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(fpr, tpr, lw=2, label=f"{nombre} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")

    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")

    return roc_auc
