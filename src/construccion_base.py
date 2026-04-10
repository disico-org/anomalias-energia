"""
Construcción de la base de datos para detección de anomalías EMCALI.

Genera archivos en data/EMCALI/datos_procesados/:
  - consumo_010924_010625.csv       : consumos con info comercial
  - info_anomalos.csv               : inspecciones lv_lega con marca anomalo
  - info_inspecciones.csv           : inspecciones realizadas con marca anomalo
  - info_inspecciones_integrada.csv : integración de ambas fuentes
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "EMCALI"
OUT_DIR = DATA_DIR / "datos_procesados"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Nombres de columnas
# ---------------------------------------------------------------------------
COL_COMERCIAL = [
    "lv_datcom_id_datcom", "producto", "contrato", "nombre_de_usuario",
    "direccion_oficial", "direccion_de_instalacion", "identificacion_direccion",
    "fecha_de_instalacion", "tipo_producto", "categoria", "subcategoria",
    "localidad", "barrio", "plan_de_facturacion", "estado_de_corte",
    "cuentas_energia", "saldo_energia", "creacion_producto", "retiro_de_producto",
    "ciclo", "nombre_usuario_oficial", "direccion", "cuentas_total",
    "saldo_total", "fecha_cargue",
]

COL_CONSUMO = [
    "lv_cons_id_cons", "ano", "mes", "consumo", "producto",
    "lv_fecha_server", "lv_user",
]

# ---------------------------------------------------------------------------
# Criterio de no-anomalía por esquema de columnas
#
# Ambas fuentes comparten la misma lógica de negocio:
#   anomalo = 0  →  irregularidad sin impacto detectable en la serie de tiempo
#                   (sellos, bornera, reconexión no autorizada, tarifa inadecuada)
#   excluir      →  categorías demasiado vagas para etiquetar
#
# Esquema IRREGULARIDAD_1 (reporte nuevo)
# ---------------------------------------------------------------------------
IRREG1_NO_ANOMALAS = [
    "16G - MEDIDOR SIN SELLO DE BORNERA Y/O ROTOS",
    "15 - RECONEXIÓN NO AUTORIZADA REGISTRADA POR EL MEDIDOR",
    "14 - CAMBIO DE USO O TARIFA INADECUADA AL TIPO DE CLIENTE",
]
IRREG1_EXCLUIR = ["16 - OTRO"]

# Esquema Anomalia 1 (lv_lega_insp / INSPECCIONES_REALIZADAS)
# Equivalencias:
#   D_S5  / D_M19  →  16G  (sello/bornera)
#   D_A6           →  15   (reconexión no autorizada)
#   CO - CONFORME  →  sin irregularidad
#   NA - NO APLICA →  16 - OTRO (excluir)
ANOMALIA1_NO_ANOMALAS = [
    "D_S5 - SIN SELLO BORNERA",
    "D_M19 - MEDIDOR SIN TAPA BORNERA",
    "D_A6 - RECONEXIÓN NO AUTORIZADA REGISTRADA POR EL MEDIDOR",
    "CO - CONFORME",
]
ANOMALIA1_EXCLUIR = ["NA - NO APLICA"]

# ---------------------------------------------------------------------------
# 1. Consumos + datos comerciales  →  consumo_010924_010625.csv
# ---------------------------------------------------------------------------
def construir_consumo() -> pd.DataFrame:
    print("Cargando datos comerciales...")
    df_com = pd.read_csv(DATA_DIR / "lv_data_comercial.csv", low_memory=False)
    df_com.columns = COL_COMERCIAL

    print("Cargando datos de consumo...")
    df_cons = pd.read_csv(DATA_DIR / "lv_data_consumo.csv", low_memory=False)
    df_cons.columns = COL_CONSUMO

    df_cons.sort_values(by=["producto", "ano", "mes"], inplace=True)

    # Crear columna fecha a partir de año y mes
    df_cons["fecha"] = pd.to_datetime(
        df_cons.rename(columns={"ano": "year", "mes": "month"})[["year", "month"]]
        .assign(day=1)
    )

    # Merge con datos comerciales
    cols_comercial = ["producto", "localidad", "barrio", "tipo_producto", "categoria", "subcategoria"]
    df = df_cons.merge(df_com[cols_comercial], on="producto", how="left", indicator=True)
    df.drop(columns=["lv_fecha_server", "lv_user"], inplace=True)

    print(f"Merge: {df._merge.value_counts().to_dict()}")

    # Separar registros sin información comercial
    df_sin_comercial = df[df._merge == "left_only"].copy()
    df_sin_comercial.drop(columns=["_merge"], inplace=True)
    df_sin_comercial.to_csv(DATA_DIR / "lv_data_consumo_sin_comercial.csv", index=False)
    print(f"Sin comercial: {len(df_sin_comercial)} registros → lv_data_consumo_sin_comercial.csv")

    # Conservar solo los que cruzaron
    df = df[df._merge == "both"].copy()
    df.drop(columns=["_merge"], inplace=True)

    fecha_min = df.fecha.min().strftime("%d%m%y")
    fecha_max = df.fecha.max().strftime("%d%m%y")
    out_path = OUT_DIR / f"consumo_{fecha_min}_{fecha_max}.csv"
    df.to_csv(out_path, index=False)
    print(f"Consumo procesado: {len(df)} registros → {out_path.name}")
    print(f"  Rango fechas: {df.fecha.min().date()} a {df.fecha.max().date()}")
    return df


# ---------------------------------------------------------------------------
# 2. Anomalías (lv_lega_insp_irregularidad)  →  info_anomalos.csv
# ---------------------------------------------------------------------------
COLS_INSP = [
    "Fecha Visita", "Producto", "Direccion",
    "Anomalia 1", "Anomalia 2", "Anomalia 3", "Anomalia 4",
    "Actividad Comercial",
]
RENAME_INSP = {"Fecha Visita": "fecha_visita", "Producto": "producto", "Direccion": "direccion"}


def construir_anomalos() -> pd.DataFrame:
    print("\nCargando inspecciones de irregularidades (lv_lega)...")
    df_raw = pd.read_excel(DATA_DIR / "lv_lega_insp_irregularidad_report02102025.xlsx")
    df_raw.drop_duplicates(inplace=True)

    df = df_raw[df_raw["Producto"] != 0][COLS_INSP].copy()
    df.rename(columns=RENAME_INSP, inplace=True)
    df["fecha_visita"] = pd.to_datetime(df["fecha_visita"], dayfirst=True)

    # Aplicar criterio de no-anomalía (equivalente a IRREGULARIDAD_1)
    df["anomalo"] = 1
    df.loc[df["Anomalia 1"].isin(ANOMALIA1_NO_ANOMALAS), "anomalo"] = 0
    df = df[~df["Anomalia 1"].isin(ANOMALIA1_EXCLUIR)].copy()

    df.sort_values(by=["producto", "fecha_visita"], inplace=True)

    out_path = OUT_DIR / "info_anomalos.csv"
    df.to_csv(out_path, index=False)
    print(f"Anomalos lv_lega: {len(df)} registros → {out_path.name}")
    print(f"  anomalo: {df.anomalo.value_counts(dropna=False).to_dict()}")
    return df


# ---------------------------------------------------------------------------
# 3. Inspecciones (INSPECCIONES_REALIZADAS)  →  info_inspecciones.csv
# ---------------------------------------------------------------------------
def construir_inspecciones() -> pd.DataFrame:
    print("\nCargando inspecciones realizadas...")
    df_raw = pd.read_excel(DATA_DIR / "INSPECCIONES_REALIZADAS.xlsx")
    df_raw.drop_duplicates(inplace=True)

    df = df_raw[df_raw["Producto"] != 0][COLS_INSP].copy()
    df.rename(columns=RENAME_INSP, inplace=True)
    df["fecha_visita"] = pd.to_datetime(df["fecha_visita"], format="mixed")

    # Aplicar criterio de no-anomalía (equivalente a IRREGULARIDAD_1)
    df["anomalo"] = 1
    df.loc[df["Anomalia 1"].isin(ANOMALIA1_NO_ANOMALAS), "anomalo"] = 0
    df = df[~df["Anomalia 1"].isin(ANOMALIA1_EXCLUIR)].copy()

    df.sort_values(by=["producto", "fecha_visita"], inplace=True)

    out_path = OUT_DIR / "info_inspecciones.csv"
    df.to_csv(out_path, index=False)
    print(f"Inspecciones: {len(df)} registros → {out_path.name}")
    print(f"  anomalo: {df.anomalo.value_counts(dropna=False).to_dict()}")
    return df


# ---------------------------------------------------------------------------
# 4. Integración de ambas fuentes  →  info_inspecciones_integrada.csv
#
# Unifica lv_lega_insp (Anomalia 1) e INSPECCIONES_REALIZADAS (Anomalia 1)
# en un esquema común, aplicando el mismo criterio de anomalía en ambas.
# Si un producto aparece en las dos fuentes se conservan ambos registros
# (pueden corresponder a fechas distintas). Los duplicados exactos se eliminan.
# ---------------------------------------------------------------------------
def integrar_inspecciones(
    df_lv_lega: pd.DataFrame | None = None,
    df_insp: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Integra los resultados de anomalías de lv_lega_insp e INSPECCIONES_REALIZADAS.

    Si no se pasan los DataFrames los construye desde los archivos en DATA_DIR.
    Devuelve un DataFrame con columnas:
        producto, fecha_visita, irregularidad_principal, anomalo, fuente
    y guarda info_inspecciones_integrada.csv en OUT_DIR.
    """
    if df_lv_lega is None:
        df_lv_lega = construir_anomalos()
    if df_insp is None:
        df_insp = construir_inspecciones()

    def _normalizar(df: pd.DataFrame, fuente: str) -> pd.DataFrame:
        out = pd.DataFrame({
            "producto":               df["producto"],
            "fecha_visita":           df["fecha_visita"],
            "irregularidad_principal": df["Anomalia 1"],
            "anomalo":                df["anomalo"],
            "fuente":                 fuente,
        })
        return out

    df_lv = _normalizar(df_lv_lega, "lv_lega")
    df_ir = _normalizar(df_insp,    "inspecciones_realizadas")

    integrado = (
        pd.concat([df_lv, df_ir], ignore_index=True)
        .drop_duplicates(subset=["producto", "fecha_visita", "irregularidad_principal"])
        .sort_values(["producto", "fecha_visita"])
        .reset_index(drop=True)
    )

    out_path = OUT_DIR / "info_inspecciones_integrada.csv"
    integrado.to_csv(out_path, index=False)

    print(f"\nIntegración:")
    print(f"  lv_lega:                 {len(df_lv):>6} registros")
    print(f"  inspecciones_realizadas: {len(df_ir):>6} registros")
    print(f"  integrado (sin dupes):   {len(integrado):>6} registros")
    print(f"  anomalo: {integrado.anomalo.value_counts(dropna=False).to_dict()}")
    print(f"  → {out_path.name}")
    return integrado


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    construir_consumo()
    df_lv   = construir_anomalos()
    df_insp = construir_inspecciones()
    integrar_inspecciones(df_lv, df_insp)
    print("\nListo. Archivos guardados en:", OUT_DIR)
