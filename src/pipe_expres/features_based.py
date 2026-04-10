from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
from tsfresh import extract_features
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import numpy as np

import pandas as pd
from tsfresh.feature_extraction import (
    ComprehensiveFCParameters,
    extract_features,
)
from typing import Iterable, Sequence

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_FEATURES: Sequence[str] = [
    "number_peaks",
    "sample_entropy",
    "skewness",
    "kurtosis",
    "percentage_of_reoccurring_values_to_all_values",
    "change_quantiles",
    "autocorrelation",
    "linear_trend",
    "cid_ce",
    "mean_second_derivative_central",
    "absolute_sum_of_changes",
    "longest_streak_below_mean",
    "first_location_of_max",
    "fft_coefficient__coeff_0__attr_abs",
    "variation_coefficient",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_consumption_features(
    df: pd.DataFrame,
    *,
    consumption_col: str = "consumo",
    id_col: str = "CLIENTE_ID",
    date_col: str = "fecha",
    group_vars: Iterable[str] | None = None,
    features_selected: Sequence[str] | None = None,
    n_jobs: int = 4,
) -> Tuple[pd.DataFrame, List[str]]:
    """Extract statistical and frequency‑domain features from a consumption time‑series.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing at least ``id_col``, ``date_col`` and ``consumption_col``.
        All additional columns listed in *group_vars* are preserved as metadata.
    consumption_col : str, default ``"consumo"``
        Column that stores the target numeric series.
    id_col : str, default ``"CLIENTE_ID"``
        Column that uniquely identifies each entity (e.g. customer).
    date_col : str, default ``"fecha"``
        Timestamp column indicating the observation date.
    group_vars : Iterable[str] or None, optional
        Extra categorical columns to attach to the resulting feature table.
    features_selected : Sequence[str] or None, optional
        Subset of tsfresh feature names to compute. If *None*, ``DEFAULT_FEATURES`` are used.
    n_jobs : int, default ``4``
        Number of parallel workers during feature extraction.

    Returns
    -------
    df_full : pandas.DataFrame
        The final table with extracted features and metadata (one row per ``id_col``).
    feature_cols : list[str]
        Names of the generated feature columns (useful for downstream modelling).

    Notes
    -----
    The function caches the feature‑calculator configuration in memory between calls so
    that subsequent executions with the same *features_selected* incur virtually no
    overhead during setup.

    Examples
    --------
    >>> df_full, feature_cols = extract_consumption_features(
    ...     df,
    ...     consumption_col="consumo",
    ...     id_col="CLIENTE_ID",
    ...     date_col="fecha",
    ...     group_vars=["BARRIO", "ESTRATO", "D_CLASE_SERVICIO_MES"],
    ...     n_jobs=8,
    ... )
    >>> df_full.shape
    (3500, 19)
    >>> feature_cols[:5]
    ['number_peaks', 'sample_entropy', 'skewness', 'kurtosis', 'percentage_of_reoccurring_values_to_all_values']
    """

    # Use provided feature list or default one
    if features_selected is None:
        features_selected = DEFAULT_FEATURES

    # ---------------------------------------------------------------------
    # 1. Convert timestamp column to ordinal days (vectorised).
    # ---------------------------------------------------------------------
    df = df.copy()
    ts = pd.to_datetime(df[date_col], errors="coerce")
    df["fecha_ordinal"] = (ts.view("int64") // 86_400_000_000_000).astype("int32")
    # Explanation: converts nanoseconds -> days since epoch (1970‑01‑01).

    # ---------------------------------------------------------------------
    # 2. Configure tsfresh calculators (cached for reuse).
    # ---------------------------------------------------------------------
    cache_key = tuple(sorted(features_selected))
    if not hasattr(extract_consumption_features, "_setting_cache"):
        extract_consumption_features._setting_cache = {}
    setting_cache = extract_consumption_features._setting_cache  # type: ignore[attr-defined]

    if cache_key in setting_cache:
        selected_settings = setting_cache[cache_key]
    else:
        settings = ComprehensiveFCParameters()
        selected_settings = {
            k: v
            for k, v in settings.items()
            if any(k.startswith(fs.split("__")[0]) for fs in features_selected)
        }
        setting_cache[cache_key] = selected_settings

    # ---------------------------------------------------------------------
    # 3. Feature extraction (parallelised).
    # ---------------------------------------------------------------------
    df_features = (
        extract_features(
            df[[id_col, "fecha_ordinal", consumption_col]],
            column_id=id_col,
            column_sort="fecha_ordinal",
            column_value=consumption_col,
            default_fc_parameters=selected_settings,
            n_jobs=n_jobs,
        )
        .dropna(axis=1, how="any")
        .astype("float32")
    )

    # ---------------------------------------------------------------------
    # 4. Attach metadata (if any).
    # ---------------------------------------------------------------------
    if group_vars:
        df_metadata = df[[id_col, *group_vars]].drop_duplicates(subset=id_col)
        df_full = df_features.merge(df_metadata, left_index=True, right_on=id_col)
    else:
        df_full = df_features.reset_index(drop=False)

    return df_full, df_features.columns.tolist()




def detectar_outliers_por_grupo(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    group_cols: Iterable[str] = ("BARRIO", "ESTRATO", "D_CLASE_SERVICIO_MES"),
    *,
    min_group_size: int = 30,
    contamination: float = 0.05,
    return_only_anomalies: bool = True,
    random_state: int | None = 42,
) -> pd.DataFrame:
    """
    Detecta observaciones atípicas *dentro* de cada subgrupo del DataFrame
    utilizando ``IsolationForest``.

    Parameters
    ----------
    df :
        DataFrame de entrada.
    feature_cols :
        Columnas numéricas que servirán como variables explicativas.
    group_cols :
        Columnas que definen los subgrupos.  
        Por defecto: ``("BARRIO", "ESTRATO", "D_CLASE_SERVICIO_MES")``.
    min_group_size :
        Tamaño mínimo del subgrupo para aplicar el modelo.  
        Subgrupos más pequeños se omiten (valor por defecto 10).
    contamination :
        Fracción estimada de outliers dentro de cada subgrupo (0 < *contamination* ≤ 0.5).
    return_only_anomalies :
        • ``True``  → devuelve **solo** las filas anómalas.  
        • ``False`` → devuelve todas las filas con la columna ``"anomaly_score"`` (0 = normal, 1 = atípico).
    random_state :
        Fija la semilla del generador aleatorio para reproducibilidad.

    Returns
    -------
    pd.DataFrame
        DataFrame con la marca ``"anomaly_score"``.  Si ``return_only_anomalies`` es
        ``True``, incluye únicamente los registros detectados como atípicos.

    Notes
    -----
    * Cada subgrupo se escala individualmente (``StandardScaler``) antes de entrenar
      ``IsolationForest``; así se respeta la escala local de las variables.
    * Para acelerar en conjuntos muy grandes puedes paralelizar el bucle de subgrupos
      con ``joblib.Parallel`` o ``concurrent.futures``—dejé el *hook* en ``n_jobs=None``.
    """
    resultados: list[pd.DataFrame] = []

    # Recorre cada subgrupo
    for _, grupo in df.groupby(list(group_cols)):
        if len(grupo) < min_group_size:
            continue

        # --- Preprocesamiento ---
        X = grupo[list(feature_cols)].replace([np.inf, -np.inf], np.nan).fillna(0)
        X_scaled = StandardScaler().fit_transform(X)

        # --- Modelo ---
        modelo = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_jobs=None,
        )
        pred = modelo.fit_predict(X_scaled)
        scores = modelo.decision_function(X_scaled)
        
        # Invertimos el score para que 1 sea lo más anómalo
        scaled_scores = MinMaxScaler().fit_transform(-scores.reshape(-1, 1)).flatten()

        # --- Mapea predicciones ---
        grupo = grupo.copy()  # evita SettingWithCopyWarning
        grupo["anomaly_score"] = scores
        grupo["anomaly_score_scaled"] = scaled_scores
        
        grupo["anomaly_bin"] = (pred == -1).astype(int)  # 1 = atípico
        resultados.append(grupo)

    if not resultados:
        raise ValueError(
            "Ningún subgrupo cumple el tamaño mínimo especificado "
            f"({min_group_size}). Ajusta 'min_group_size' o revisa tus datos."
        )

    df_out = pd.concat(resultados, ignore_index=True)

    if return_only_anomalies:
        df_out = df_out[df_out["anomaly_bin"] == 1]

    return df_out
