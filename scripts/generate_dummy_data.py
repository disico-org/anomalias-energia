"""
Genera o limpia datos de ejemplo para el dashboard DISICO.

Lee los 2 Excel existentes (top_5000_supervisado.xlsx, top_5000_no_supervisado.xlsx)
para extraer IDs de producto y genera los 5 archivos complementarios que el dashboard
necesita para mostrar todas las secciones.

Comandos:
    uv run python scripts/generate_dummy_data.py generate
    uv run python scripts/generate_dummy_data.py generate --data-path /app/data
    uv run python scripts/generate_dummy_data.py clean
    uv run python scripts/generate_dummy_data.py clean --data-path /app/data
    uv run python scripts/generate_dummy_data.py status
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# Archivos que este script genera (y puede limpiar)
GENERATED_FILES = [
    "consumo_011223_01062025_filtrado.parquet",
    "georeferencias.csv",
    "informacion_curva_roc.pickle",
    "resultados_1.parquet",
    "resultados_df_last.csv",
]

MARKER_FILE = ".dummy_data_marker.json"


def _resolve_data_path(args_path: str | None) -> Path:
    if args_path:
        return Path(args_path)
    return Path(__file__).resolve().parent.parent / "data"


def _write_marker(data_path: Path):
    from datetime import datetime, timezone
    marker = data_path / MARKER_FILE
    info = {
        "tipo": "dummy_data",
        "generado": datetime.now(timezone.utc).isoformat(),
        "archivos": GENERATED_FILES,
        "nota": "Datos de ejemplo. Ejecutar 'clean' para eliminarlos.",
    }
    marker.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")


def _has_marker(data_path: Path) -> bool:
    return (data_path / MARKER_FILE).exists()


# ── Funciones de generación (vectorizadas) ───────────────────

def extract_ids(data_path: Path) -> np.ndarray:
    ids = set()
    for name, id_candidates in [
        ("top_5000_supervisado.xlsx", ["producto"]),
        ("top_5000_no_supervisado.xlsx", ["producto", "CLIENTE_ID", "CLIENTES_ID", "cliente_id"]),
    ]:
        fpath = data_path / name
        if not fpath.exists():
            print(f"  ⚠ No se encontró {fpath}, omitiendo.")
            continue
        df = pd.read_excel(fpath, engine="openpyxl")
        col = next((c for c in id_candidates if c in df.columns), df.columns[1])
        vals = pd.to_numeric(df[col], errors="coerce").dropna().astype(int).unique()
        ids.update(vals)
        print(f"  ✓ {name}: {len(vals)} IDs (col: {col})")
    if not ids:
        raise RuntimeError("No se pudo extraer ningún ID de los Excel.")
    return np.array(sorted(ids))


def assign_demographics(ids: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    n = len(ids)
    return pd.DataFrame({
        "CLIENTE_ID": ids,
        "localidad": rng.choice(["1", "2", "3", "4", "5"], size=n, p=[0.35, 0.25, 0.20, 0.12, 0.08]),
        "barrio": rng.choice([str(i) for i in range(101, 121)], size=n),
        "tipo_producto": rng.choice(["1", "2", "3"], size=n, p=[0.60, 0.30, 0.10]),
        "categoria": rng.choice(["1", "2", "3", "4", "5", "6"], size=n, p=[0.30, 0.25, 0.20, 0.12, 0.08, 0.05]),
        "subcategoria": rng.choice(["1", "2", "3", "4"], size=n, p=[0.40, 0.30, 0.20, 0.10]),
    })


def gen_consumo(ids: np.ndarray, demo: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Genera serie temporal mensual (vectorizado, sin loops Python)."""
    fechas = pd.date_range("2023-12-01", "2025-06-01", freq="MS")
    n_meses = len(fechas)
    n_ids = len(ids)

    # Matrices vectorizadas: (n_ids, n_meses)
    base = rng.lognormal(mean=5.5, sigma=0.5, size=n_ids)
    noise = rng.normal(0, 0.15, size=(n_ids, n_meses))
    seasonal = 0.1 * np.sin(2 * np.pi * np.arange(n_meses) / 12)
    consumo_matrix = base[:, None] * np.exp(noise + seasonal[None, :])
    consumo_matrix = np.clip(consumo_matrix, 20, 2000).round(2)

    # Construir DataFrame con repeat/tile (sin loop)
    ids_rep = np.repeat(ids, n_meses)
    fechas_tile = np.tile(fechas, n_ids)
    consumo_flat = consumo_matrix.ravel()

    # consumo_prev: shift por cliente (columna anterior en la matriz)
    prev_matrix = np.column_stack([consumo_matrix[:, 0:1] * 0.95, consumo_matrix[:, :-1]]).round(2)

    df = pd.DataFrame({
        "CLIENTE_ID": ids_rep,
        "fecha": fechas_tile,
        "consumo": consumo_flat,
        "mes": pd.DatetimeIndex(fechas_tile).month,
        "consumo_actual": consumo_flat,
        "consumo_prev": prev_matrix.ravel(),
    })
    return df.merge(demo, on="CLIENTE_ID", how="left")


def gen_georeferencias(ids: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    n = len(ids)
    lat = 3.4516 + rng.normal(0, 0.03, size=n)
    lon = -76.5320 + rng.normal(0, 0.03, size=n)
    mask = rng.random(n) < 0.05
    lat[mask] = np.nan
    lon[mask] = np.nan
    return pd.DataFrame({"producto": ids, "lat": lat, "lon": lon})


def gen_roc() -> tuple:
    fpr = np.linspace(0, 1, 200)
    tpr = 1 - (1 - fpr) ** 2.5
    tpr = tpr * 0.712 / np.trapz(tpr, fpr)
    tpr = np.clip(tpr, 0, 1)
    tpr[0], tpr[-1] = 0.0, 1.0
    return (fpr, tpr)


def gen_resultados(ids: np.ndarray, demo: pd.DataFrame, rng: np.random.Generator):
    """Genera resultados_1 y resultados_df_last (vectorizado)."""
    fechas = pd.date_range("2023-12-01", "2025-06-01", freq="MS")
    n_meses = len(fechas)
    n_ids = len(ids)

    # Vectorizado: matrices (n_ids, n_meses)
    base = rng.lognormal(5.5, 0.5, size=n_ids)
    prev_matrix = (base[:, None] * rng.lognormal(0, 0.15, size=(n_ids, n_meses))).round(2)
    actual_matrix = (base[:, None] * rng.lognormal(0, 0.15, size=(n_ids, n_meses))).round(2)

    ids_rep = np.repeat(ids, n_meses)
    fechas_tile = np.tile(fechas.strftime("%Y-%m-%d"), n_ids)

    df_r1 = pd.DataFrame({
        "CLIENTE_ID": ids_rep,
        "consumo_prev": prev_matrix.ravel(),
        "consumo_actual": actual_matrix.ravel(),
        "year_month": fechas_tile,
    })
    df_r1 = df_r1.merge(demo, on="CLIENTE_ID", how="left")

    # df_last: un registro por cliente (último mes)
    df_last = demo.copy()
    df_last["fraud_score"] = np.round(rng.beta(2, 5, size=n_ids), 6)
    df_last["year_month"] = fechas[-1].strftime("%Y-%m-%d")

    return df_r1, df_last


# ── Comandos ─────────────────────────────────────────────────

def cmd_generate(data_path: Path):
    print("=" * 60)
    print("DISICO — Generador de Datos de Ejemplo")
    print("=" * 60)
    print(f"Ruta de datos: {data_path}\n")

    if _has_marker(data_path):
        print("⚠ Ya existen datos de ejemplo. Se sobreescribirán.\n")

    existing_real = []
    for f in GENERATED_FILES:
        if (data_path / f).exists() and not _has_marker(data_path):
            existing_real.append(f)
    if existing_real:
        print("⛔ Se encontraron archivos que NO fueron generados por este script:")
        for f in existing_real:
            print(f"   - {f}")
        print("\nEstos podrían ser datos reales de producción.")
        print("Elimínalos manualmente o usa --force para ignorar esta verificación.")
        return False

    rng = np.random.default_rng(42)

    print("📋 Extrayendo IDs de producto de los Excel...")
    ids = extract_ids(data_path)
    print(f"   Total IDs únicos: {len(ids)}\n")

    demo = assign_demographics(ids, rng)

    print("📊 Generando serie temporal de consumo...")
    df_consumo = gen_consumo(ids, demo, rng)
    out = data_path / "consumo_011223_01062025_filtrado.parquet"
    df_consumo.to_parquet(out, index=False)
    print(f"   ✓ {out.name}: {len(df_consumo):,} filas")

    print("🗺️  Generando georeferencias (Cali)...")
    df_geo = gen_georeferencias(ids, rng)
    out = data_path / "georeferencias.csv"
    df_geo.to_csv(out, index=False)
    print(f"   ✓ {out.name}: {len(df_geo):,} filas")

    print("📈 Generando curva ROC sintética...")
    roc_data = gen_roc()
    out = data_path / "informacion_curva_roc.pickle"
    with open(out, "wb") as f:
        pickle.dump(roc_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    auc = float(np.trapz(roc_data[1], roc_data[0]))
    print(f"   ✓ {out.name}: AUC = {auc:.3f}")

    print("🧮 Generando resultados para Mahalanobis...")
    df_r1, df_last = gen_resultados(ids, demo, rng)
    out_r1 = data_path / "resultados_1.parquet"
    out_last = data_path / "resultados_df_last.csv"
    df_r1.to_parquet(out_r1, index=False)
    df_last.to_csv(out_last, index=False)
    print(f"   ✓ {out_r1.name}: {len(df_r1):,} filas")
    print(f"   ✓ {out_last.name}: {len(df_last):,} filas")

    _write_marker(data_path)

    cache_path = data_path / "cache" / "mahalanobis_cache.pkl"
    if cache_path.exists():
        cache_path.unlink()
        print("   ✓ Cache de Mahalanobis eliminado (se recalculará)")

    print("\n" + "=" * 60)
    print("✅ Datos de ejemplo generados!")
    print("=" * 60)
    for f in GENERATED_FILES:
        p = data_path / f
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  {f}: {size_mb:.2f} MB")
    return True


def cmd_clean(data_path: Path, force: bool = False):
    print("=" * 60)
    print("DISICO — Limpieza de Datos de Ejemplo")
    print("=" * 60)
    print(f"Ruta de datos: {data_path}\n")

    if not _has_marker(data_path) and not force:
        print("⚠ No se encontró el marcador de datos de ejemplo.")
        print("  Los archivos actuales podrían ser datos reales.")
        print("  Usa --force si estás seguro de que deseas eliminarlos.")
        return False

    removed = []
    for f in GENERATED_FILES:
        p = data_path / f
        if p.exists():
            p.unlink()
            removed.append(f)

    cache_path = data_path / "cache" / "mahalanobis_cache.pkl"
    if cache_path.exists():
        cache_path.unlink()
        removed.append("cache/mahalanobis_cache.pkl")

    marker = data_path / MARKER_FILE
    if marker.exists():
        marker.unlink()

    print("Archivos eliminados:")
    for f in removed:
        print(f"  ✓ {f}")

    print("\n✅ Limpieza completada!")
    print("El dashboard ahora usará datos reales si se encuentran en esta ruta.")
    return True


def cmd_status(data_path: Path):
    print("=" * 60)
    print("DISICO — Estado de Datos")
    print("=" * 60)
    print(f"Ruta de datos: {data_path}\n")

    if _has_marker(data_path):
        marker = data_path / MARKER_FILE
        info = json.loads(marker.read_text(encoding="utf-8"))
        print(f"📌 MODO: Datos de ejemplo")
        print(f"   Generados: {info.get('generado', 'desconocido')}")
    else:
        print(f"📌 MODO: Datos reales (o sin datos)")

    print("\nArchivos:")
    all_files = GENERATED_FILES + [
        "top_5000_supervisado.xlsx", "top_5000_no_supervisado.xlsx",
        "cache/mahalanobis_cache.pkl",
    ]
    for f in all_files:
        p = data_path / f
        if p.exists():
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  ✓ {f} ({size_mb:.2f} MB)")
        else:
            print(f"  ✗ {f}")


def main():
    parser = argparse.ArgumentParser(
        description="Genera o limpia datos de ejemplo para el dashboard DISICO"
    )
    parser.add_argument(
        "command", choices=["generate", "clean", "status"],
        help="generate | clean | status"
    )
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data_path = _resolve_data_path(args.data_path)
    if not data_path.exists():
        print(f"⛔ La carpeta de datos no existe: {data_path}")
        return

    if args.command == "generate":
        if args.force:
            _write_marker(data_path)
        cmd_generate(data_path)
    elif args.command == "clean":
        cmd_clean(data_path, force=args.force)
    elif args.command == "status":
        cmd_status(data_path)


if __name__ == "__main__":
    main()
