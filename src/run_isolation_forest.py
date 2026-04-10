"""
run_isolation_forest.py
-----------------------
Pipeline standalone para detección de anomalías con Isolation Forest
sobre features extraídas con tsfresh. Incluye preprocesamiento y
checkpointing en cada paso.

Uso:
    python src/run_isolation_forest.py
    python src/run_isolation_forest.py --sample 2000   # prueba rápida con 2000 clientes

Salidas en data/Resultados/:
    clientes_series_invalidas.csv          (compartido con otros métodos)
    if_batches/batch_0000.parquet ...      (features tsfresh por batch, se van acumulando)
    if_df_full.parquet                     (features tsfresh consolidadas, un row por cliente)
    if_feature_cols.json                   (lista de columnas de features para retomar desde paso 2)
    if_resultados_raw.parquet              (scores de outlier por cliente)
    if_resultados_final.csv               (ranking final ordenado por anomaly_score)
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Rutas base (nivel módulo: seguro para procesos hijos de multiprocessing)
BASE = Path(__file__).parent.parent
DATA = BASE / "data" / "EMCALI" / "datos_procesados"
OUT = BASE / "data" / "Resultados"

sys.path.insert(0, str(Path(__file__).parent))

parser = argparse.ArgumentParser()
parser.add_argument("--sample", type=int, default=None,
                    help="Número de clientes a usar (para pruebas rápidas)")
parser.add_argument("--batch-size", type=int, default=10_000,
                    help="Clientes por batch en extracción tsfresh (default: 10000)")


if __name__ == "__main__":
    from pipe_expres.features_based import detectar_outliers_por_grupo, extract_consumption_features
    from pipe_expres.limpieza import filtrar_clientes
    from pipe_expres.series_anomalas import get_no_validas

    args = parser.parse_args()
    SAMPLE = args.sample
    BATCH_SIZE = args.batch_size
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
    # PASO 1: Extracción de features con tsfresh por batch (con checkpoint)
    # ---------------------------------------------------------------------------

    DF_FULL_PATH = OUT / f"{PREFIX}if_df_full.parquet"
    FEATURE_COLS_PATH = OUT / f"{PREFIX}if_feature_cols.json"
    BATCHES_DIR = OUT / f"{PREFIX}if_batches"

    if DF_FULL_PATH.exists() and FEATURE_COLS_PATH.exists():
        print(f"\n[PASO 1] Cargando features consolidadas desde {DF_FULL_PATH}")
        df_full = pd.read_parquet(DF_FULL_PATH, engine="pyarrow")
        with open(FEATURE_COLS_PATH) as f:
            feature_cols = json.load(f)
        print(f"  → {len(feature_cols)} columnas de features cargadas")
    else:
        BATCHES_DIR.mkdir(parents=True, exist_ok=True)
        todos_clientes = df_filtrado.CLIENTE_ID.unique()
        n_batches = (len(todos_clientes) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"\n[PASO 1] Extrayendo features con tsfresh en {n_batches} batches de {BATCH_SIZE:,} clientes (n_jobs=8)...")

        feature_cols = None
        for i in range(n_batches):
            batch_path = BATCHES_DIR / f"batch_{i:04d}.parquet"
            if batch_path.exists():
                print(f"  [batch {i+1}/{n_batches}] Ya existe, saltando.")
                continue

            clientes_batch = todos_clientes[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
            df_batch = df_filtrado[df_filtrado.CLIENTE_ID.isin(clientes_batch)]
            print(f"  [batch {i+1}/{n_batches}] Procesando {len(clientes_batch):,} clientes...")

            df_batch_full, feature_cols = extract_consumption_features(
                df_batch,
                n_jobs=8,
                group_vars=["barrio", "tipo_producto", "categoria"],
            )
            df_batch_full.to_parquet(batch_path, index=False)
            print(f"    → guardado en {batch_path}")

        # Consolidar todos los batches
        print("\n[PASO 1] Consolidando batches...")
        batch_files = sorted(BATCHES_DIR.glob("batch_*.parquet"))
        df_full = pd.concat(
            [pd.read_parquet(p, engine="pyarrow") for p in batch_files],
            ignore_index=True,
        )

        # Obtener feature_cols del primer batch si no están en memoria
        if feature_cols is None:
            sample_batch = pd.read_parquet(batch_files[0], engine="pyarrow")
            meta_cols = {"CLIENTE_ID", "barrio", "tipo_producto", "categoria"}
            feature_cols = [c for c in sample_batch.columns if c not in meta_cols]

        df_full.to_parquet(DF_FULL_PATH, index=False)
        with open(FEATURE_COLS_PATH, "w") as f:
            json.dump(feature_cols, f)
        print(f"  → df_full consolidado guardado en {DF_FULL_PATH} ({len(df_full):,} clientes)")
        print(f"  → {len(feature_cols)} feature columns guardadas en {FEATURE_COLS_PATH}")

    print(f"[PASO 1] Shape df_full: {df_full.shape}")

    # ---------------------------------------------------------------------------
    # PASO 2: Isolation Forest por grupo (con checkpoint)
    # ---------------------------------------------------------------------------

    RAW_PATH = OUT / f"{PREFIX}if_resultados_raw.parquet"
    GROUP_COLS = ["barrio", "tipo_producto", "categoria"]

    if RAW_PATH.exists():
        print(f"\n[PASO 2] Cargando resultados crudos desde {RAW_PATH}")
        resultados_raw = pd.read_parquet(RAW_PATH, engine="pyarrow")
    else:
        print("\n[PASO 2] Ejecutando Isolation Forest por grupo...")
        resultados_raw = detectar_outliers_por_grupo(
            df=df_full,
            feature_cols=feature_cols,
            group_cols=GROUP_COLS,
            min_group_size=25,
            contamination=0.05,
            return_only_anomalies=False,
        )
        resultados_raw.to_parquet(RAW_PATH, index=False)
        print(f"  → {len(resultados_raw):,} registros guardados en {RAW_PATH}")

    n_anomalos = (resultados_raw["anomaly_bin"] == 1).sum() if "anomaly_bin" in resultados_raw.columns else "N/A"
    print(f"[PASO 2] Shape resultados_raw: {resultados_raw.shape}")
    print(f"[PASO 2] Clientes marcados como anómalos: {n_anomalos:,}" if isinstance(n_anomalos, int) else f"[PASO 2] anomaly_bin: {n_anomalos}")

    # ---------------------------------------------------------------------------
    # PASO 3: Ordenar y guardar ranking final
    # ---------------------------------------------------------------------------

    print("\n[PASO 3] Generando ranking final...")

    resultados_final = resultados_raw.sort_values(by="anomaly_score", ascending=False)

    FINAL_PATH = OUT / f"{PREFIX}if_resultados_final.csv"
    resultados_final.to_csv(FINAL_PATH, index=False)

    print(f"[PASO 3] Ranking final guardado en {FINAL_PATH}")
    print(f"  → {len(resultados_final):,} clientes rankeados")
    print(f"  → anomaly_score máximo: {resultados_final['anomaly_score'].max():.4f}")
    print(f"  → anomaly_score mediano: {resultados_final['anomaly_score'].median():.4f}")
    print("\nPipeline Isolation Forest completado.")
