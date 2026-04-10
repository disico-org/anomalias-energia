"""
Script de pre-cálculo de Mahalanobis para DISICO Dashboard.

Este script debe ejecutarse ANTES del deploy para generar el cache
y reducir el tiempo de inicio del dashboard de ~10 minutos a ~30 segundos.

Uso:
    python scripts/precompute_mahalanobis.py

Salida:
    data/cache/mahalanobis_cache.pkl - Cache con cálculos pre-hechos
"""

import os
import sys
import pickle
from pathlib import Path
import numpy as np
import pandas as pd


def precompute_mahalanobis(data_path: Path, cache_path: Path):
    """
    Pre-calcula las distancias de Mahalanobis para todos los clientes.
    
    Parameters
    ----------
    data_path : Path
        Ruta base donde están los archivos de datos
    cache_path : Path
        Ruta donde guardar el cache generado
    """
    print("=" * 60)
    print("DISICO - Pre-cálculo de Mahalanobis")
    print("=" * 60)
    
    # Rutas de archivos
    file_resultados_1 = data_path / "resultados_1.parquet"
    file_resdf_last = data_path / "resultados_df_last.csv"
    
    # Verificar que existen los archivos
    if not file_resultados_1.exists():
        print(f"❌ Error: No se encuentra {file_resultados_1}")
        sys.exit(1)
    
    if not file_resdf_last.exists():
        print(f"❌ Error: No se encuentra {file_resdf_last}")
        sys.exit(1)
    
    print(f"✓ Archivos encontrados")
    print(f"  - resultados_1.parquet")
    print(f"  - resultados_df_last.csv")
    
    # Cargar datos
    print("\n📊 Cargando datos...")
    try:
        df_res1 = pd.read_parquet(file_resultados_1)
        df_last = pd.read_csv(file_resdf_last)
        print(f"✓ Datos cargados: {len(df_res1):,} filas (resultados_1), {len(df_last):,} filas (last)")
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        sys.exit(1)
    
    # Normalizar columnas de fecha
    for d in [df_res1, df_last]:
        if "year_month" in d.columns:
            d["year_month"] = (
                pd.to_datetime(d["year_month"], errors="coerce")
                .dt.strftime("%Y-%m-%d")
            )
    
    # Columnas de agrupación
    gc = ["localidad", "barrio", "tipo_producto", "categoria", "subcategoria"]
    av_gc = [c for c in gc if c in df_res1.columns]
    has_ym = "year_month" in df_res1.columns and "year_month" in df_last.columns
    kc = av_gc + (["year_month"] if has_ym else [])
    
    print(f"\n🔧 Columnas de agrupación: {kc}")
    
    # Preparar datos
    need = [c for c in kc + ["CLIENTE_ID", "consumo_prev", "consumo_actual"] if c in df_res1.columns]
    r1 = df_res1[need].copy()
    
    for c in kc:
        r1[c] = r1[c].astype(str)
        df_last[c] = df_last[c].astype(str)
    
    r1["_gk"] = r1[kc].agg("||".join, axis=1)
    clean = r1.dropna(subset=["consumo_prev", "consumo_actual"])
    
    # Agrupar datos
    print("📦 Agrupando datos por categorías...")
    grouped = {
        k: (g[["consumo_prev", "consumo_actual"]].values, g["CLIENTE_ID"].values)
        for k, g in clean.groupby("_gk", sort=False)
    }
    print(f"✓ {len(grouped)} grupos creados")
    
    # Columna de fraud score
    fc = "fraud_score" if "fraud_score" in df_last.columns else None
    if fc:
        print(f"✓ Columna de fraud_score encontrada")
    
    # Calcular Mahalanobis para cada cliente
    print("\n🧮 Calculando distancias de Mahalanobis...")
    print("   (Este proceso puede tomar varios minutos)")
    
    seen = {}
    cache = {}
    total = len(df_last)
    
    for idx, row in df_last.iterrows():
        if idx % 1000 == 0:
            print(f"   Progreso: {idx:,} / {total:,} ({100*idx/total:.1f}%)", end="\r")
        
        try:
            cid = int(row["CLIENTE_ID"])
        except (ValueError, TypeError):
            continue
        
        if cid in cache:
            continue
        
        key = "||".join(str(row[c]) for c in kc)
        
        if key in seen:
            ge = seen[key]
        else:
            pair = grouped.get(key)
            if pair is None:
                continue
            
            X, ids = pair
            if X.shape[0] < 2:
                continue
            
            # Calcular estadísticas del grupo
            mu = X.mean(axis=0)
            cov = np.cov(X, rowvar=False, ddof=0)
            
            # Regularizar matriz de covarianza si es singular
            if np.linalg.det(cov) == 0:
                cov += np.eye(2) * 1e-6
            
            inv = np.linalg.inv(cov)
            md = np.sqrt(((X - mu) @ inv * (X - mu)).sum(axis=1))
            
            ge = {
                "X": X,
                "ids_group": ids,
                "mu": mu,
                "cov": cov,
                "mah_dist": md,
                "es_anomalo": md > 2,
                "filter_desc": {c: row[c] for c in kc}
            }
            seen[key] = ge
        
        cache[cid] = {
            **ge,
            "fraud_score": float(row[fc]) if fc else None
        }
    
    print(f"   Progreso: {total:,} / {total:,} (100.0%)     ")
    print(f"✓ Cálculo completado: {len(cache):,} clientes procesados")
    
    # Guardar cache
    print(f"\n💾 Guardando cache...")
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path / "mahalanobis_cache.pkl"
    
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        file_size = cache_file.stat().st_size / (1024 * 1024)  # MB
        print(f"✓ Cache guardado: {cache_file}")
        print(f"   Tamaño: {file_size:.2f} MB")
    except Exception as e:
        print(f"❌ Error guardando cache: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ Pre-cálculo completado exitosamente!")
    print("=" * 60)
    print(f"\nEl dashboard cargará en ~30 segundos en lugar de ~10 minutos.")
    print(f"Recuerda subir el archivo '{cache_file}' al servidor via SFTP.")


def main():
    """Función principal."""
    # Determinar rutas
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    
    # Usar variable de entorno si existe, o default local
    data_path = Path(os.environ.get("DATA_PATH", project_dir / "data"))
    cache_path = Path(os.environ.get("CACHE_PATH", data_path / "cache"))
    
    print(f"Ruta de datos: {data_path}")
    print(f"Ruta de cache: {cache_path}")
    
    # Verificar que existe la carpeta de datos
    if not data_path.exists():
        print(f"❌ Error: La carpeta de datos no existe: {data_path}")
        print("   Crea la carpeta 'data' y coloca los archivos necesarios:")
        print("   - resultados_1.parquet")
        print("   - resultados_df_last.csv")
        sys.exit(1)
    
    # Ejecutar pre-cálculo
    precompute_mahalanobis(data_path, cache_path)


if __name__ == "__main__":
    main()
