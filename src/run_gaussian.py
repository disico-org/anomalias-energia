"""
run_gaussian.py
---------------
Pipeline standalone para detección de anomalías por distancia de Mahalanobis
(método Gaussiano). Incluye preprocesamiento y checkpointing en cada paso.

Uso:
    python src/run_gaussian.py
    python src/run_gaussian.py --sample 2000   # prueba rápida con 2000 clientes

Salidas en data/Resultados/:
    clientes_series_invalidas.csv     (compartido con otros métodos)
    gaussian_df_features.parquet      (features intermedias)
    gaussian_resultados_raw.parquet   (scores por cliente×mes)
    gaussian_resultados_final.csv     (ranking final, un registro por cliente)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Rutas base (nivel módulo: seguro para procesos hijos de multiprocessing)
BASE = Path(__file__).parent.parent
DATA = BASE / "data" / "EMCALI" / "datos_procesados"
OUT = BASE / "data" / "Resultados"

sys.path.insert(0, str(Path(__file__).parent))

parser = argparse.ArgumentParser()
parser.add_argument("--sample", type=int, default=None,
                    help="Número de clientes a usar (para pruebas rápidas)")


if __name__ == "__main__":
    from pipe_expres.features import (
        crear_features,
        feature_consumo_log,
        feature_consumo_prev_year,
        feature_year_month,
    )
    from pipe_expres.gaussiana_method import calcular_fraud_scores
    from pipe_expres.limpieza import filtrar_clientes
    from pipe_expres.series_anomalas import get_no_validas

    args = parser.parse_args()
    SAMPLE = args.sample
    PREFIX = "sample_" if SAMPLE else ""

    OUT.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------------
    # PASO 0: Preprocesamiento (con checkpoint)
    # ---------------------------------------------------------------------------

    FILTRADO_PATH = DATA / "consumo_011223_01062025_filtrado.parquet"
    INVALIDAS_PATH = OUT / "clientes_series_invalidas.csv"

    if FILTRADO_PATH.exists():
        print(f"[PASO 0] Cargando datos filtrados desde {FILTRADO_PATH}")
        df_filtrado = pd.read_parquet(FILTRADO_PATH, engine="pyarrow")
    else:
        print("[PASO 0] Cargando datos crudos...")
        df = pd.read_parquet(DATA / "consumo_011223_01062025.parquet", engine="pyarrow")

        print("[PASO 0] Identificando series no válidas...")
        primera_seleccion = get_no_validas(df, cliente_col="producto")
        primera_seleccion["serie_no_valida"] = 1
        primera_seleccion.to_csv(INVALIDAS_PATH, index=False)
        print(f"  → {len(primera_seleccion)} series inválidas guardadas en {INVALIDAS_PATH}")

        print("[PASO 0] Filtrando clientes...")
        df["CLIENTE_ID"] = df["producto"]
        df_sin_invalidas = df[~df.CLIENTE_ID.isin(primera_seleccion.CLIENTE_ID.unique())]
        df_filtrado = filtrar_clientes(
            df_sin_invalidas,
            umbral_consumo_min=5,
            percentil_alto=0.99,
            eliminar_recientes=False,
            eliminar_altos=False,
            meses=19,
            group_cols=["localidad", "barrio", "tipo_producto", "categoria"],
        )
        df_filtrado = df_filtrado.sort_values(
            by=["CLIENTE_ID", "fecha"], ascending=[True, True]
        ).reset_index(drop=True)

        df_filtrado.to_parquet(FILTRADO_PATH, index=False)
        print(f"  → {df_filtrado.CLIENTE_ID.nunique()} clientes guardados en {FILTRADO_PATH}")

    if SAMPLE:
        clientes_muestra = df_filtrado.CLIENTE_ID.unique()[:SAMPLE]
        df_filtrado = df_filtrado[df_filtrado.CLIENTE_ID.isin(clientes_muestra)]
        print(f"[PASO 0] Modo prueba: usando {df_filtrado.CLIENTE_ID.nunique():,} clientes")
    else:
        print(f"[PASO 0] Clientes en df_filtrado: {df_filtrado.CLIENTE_ID.nunique():,}")

    # ---------------------------------------------------------------------------
    # PASO 1: Feature engineering (con checkpoint)
    # ---------------------------------------------------------------------------

    FEATURES_PATH = OUT / f"{PREFIX}gaussian_df_features.parquet"

    if FEATURES_PATH.exists():
        print(f"\n[PASO 1] Cargando features desde {FEATURES_PATH}")
        df_features = pd.read_parquet(FEATURES_PATH, engine="pyarrow")
    else:
        print("\n[PASO 1] Creando features (log, prev_year, year_month)...")
        feature_funcs = [
            feature_consumo_log,
            feature_consumo_prev_year,
            feature_year_month,
        ]
        df_features = crear_features(df_filtrado, feature_funcs)
        df_features["year_month"] = df_features["fecha"].dt.to_period("M").astype(str)

        df_features.to_parquet(FEATURES_PATH, index=False)
        print(f"  → Features guardadas en {FEATURES_PATH}")

    print(f"[PASO 1] Shape df_features: {df_features.shape}")

    # ---------------------------------------------------------------------------
    # PASO 2: Calcular fraud scores (con checkpoint)
    # ---------------------------------------------------------------------------

    RAW_PATH = OUT / f"{PREFIX}gaussian_resultados_raw.parquet"

    GROUP_COLS = ["localidad", "barrio", "tipo_producto", "categoria", "year_month"]
    VARS_CONSUMO = ["consumo_log_prev_year", "consumo_log"]

    if RAW_PATH.exists():
        print(f"\n[PASO 2] Cargando resultados crudos desde {RAW_PATH}")
        resultados_raw = pd.read_parquet(RAW_PATH, engine="pyarrow")
    else:
        print("\n[PASO 2] Calculando fraud scores (Mahalanobis)...")
        resultados_raw = calcular_fraud_scores(
            df_features,
            group_cols=GROUP_COLS,
            variables_consumo=VARS_CONSUMO,
            min_group_size=25,
            show_progress=True,
        )
        resultados_raw.to_parquet(RAW_PATH, index=False)
        print(f"  → {len(resultados_raw):,} registros guardados en {RAW_PATH}")

    print(f"[PASO 2] Shape resultados_raw: {resultados_raw.shape}")

    # ---------------------------------------------------------------------------
    # PASO 3: Post-procesamiento y ranking final
    # ---------------------------------------------------------------------------

    print("\n[PASO 3] Generando ranking final...")

    resultados_raw["year_month"] = pd.to_datetime(resultados_raw["year_month"], format="%Y-%m")
    resultados_raw = resultados_raw.sort_values(by="year_month", ascending=False)

    # Un registro por cliente: el más reciente
    resultados_final = resultados_raw.groupby("CLIENTE_ID", as_index=False).first()
    resultados_final.sort_values(by="fraud_score", ascending=False, inplace=True)
    resultados_final["rank_gaussian"] = np.arange(1, len(resultados_final) + 1)

    FINAL_PATH = OUT / f"{PREFIX}gaussian_resultados_final.csv"
    resultados_final.to_csv(FINAL_PATH, index=False)

    print(f"[PASO 3] Ranking final guardado en {FINAL_PATH}")
    print(f"  → {len(resultados_final):,} clientes rankeados")
    print(f"  → fraud_score máximo: {resultados_final['fraud_score'].max():.4f}")
    print(f"  → fraud_score mediano: {resultados_final['fraud_score'].median():.4f}")
    print("\nPipeline Gaussiano completado.")
