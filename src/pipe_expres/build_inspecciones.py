"""
build_inspecciones.py

Genera los archivos CSV:
  - info_inspecciones.csv      (fuente: INSPECCIONES_REALIZADAS.xlsx)
  - inspecciones_etiquetadas.csv (combinación con info_nuevos_anomalos)

Basado en:
  - notebooks/EMCALI/construccion_base.ipynb
  - notebooks/EMCALI/rev_inspec.ipynb
"""

import numpy as np
import pandas as pd
from pathlib import Path
from build_anomalos import build_info_nuevos_anomalos


# ---------------------------------------------------------------------------
# Rutas por defecto
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "EMCALI"
DATA_PROC = DATA_RAW / "datos_procesados"

# Irregularidades que NO se consideran anomalías en el reporte nuevo
NO_ANOMALO = [
    "16G - MEDIDOR SIN SELLO DE BORNERA Y/O ROTOS",
    "15 - RECONEXIÓN NO AUTORIZADA REGISTRADA POR EL MEDIDOR",
    "14 - CAMBIO DE USO O TARIFA INADECUADA AL TIPO DE CLIENTE",
    "16 - OTRO",
]


# ---------------------------------------------------------------------------
# info_inspecciones
# ---------------------------------------------------------------------------
def build_info_inspecciones(
    path_xlsx: str | Path = None,
    output_path: str | Path = None,
) -> pd.DataFrame:
    """
    Procesa INSPECCIONES_REALIZADAS.xlsx y genera info_inspecciones.csv.

    Columnas de salida:
        fecha_visita, producto, direccion,
        Anomalia 1, Anomalia 2, Anomalia 3, Anomalia 4,
        Actividad Comercial, anomalo

    Reglas de etiquetado:
      - anomalo = 1  por defecto
      - anomalo = 0  si Anomalia 1 == 'CO - CONFORME'
      - anomalo = NaN si Anomalia 1 == 'NA - NO APLICA'

    Parameters
    ----------
    path_xlsx   : ruta al Excel fuente. Por defecto DATA_RAW/INSPECCIONES_REALIZADAS.xlsx.
    output_path : ruta de salida del CSV. Por defecto DATA_PROC/info_inspecciones.csv.

    Returns
    -------
    DataFrame con todas las inspecciones procesadas.
    """
    if path_xlsx is None:
        path_xlsx = DATA_RAW / "INSPECCIONES_REALIZADAS.xlsx"
    if output_path is None:
        output_path = DATA_PROC / "info_inspecciones.csv"

    path_xlsx = Path(path_xlsx)
    output_path = Path(output_path)

    df = pd.read_excel(path_xlsx)
    df.drop_duplicates(inplace=True)

    df_inspec = (
        df[df["Producto"] != 0][
            [
                "Fecha Visita",
                "Producto",
                "Direccion",
                "Anomalia 1",
                "Anomalia 2",
                "Anomalia 3",
                "Anomalia 4",
                "Actividad Comercial",
            ]
        ]
        .copy()
    )

    df_inspec["anomalo"] = 1
    df_inspec.rename(
        columns={
            "Fecha Visita": "fecha_visita",
            "Producto": "producto",
            "Direccion": "direccion",
        },
        inplace=True,
    )
    df_inspec["fecha_visita"] = pd.to_datetime(df_inspec["fecha_visita"], format="mixed")

    df_inspec.loc[df_inspec["Anomalia 1"] == "CO - CONFORME", "anomalo"] = 0
    df_inspec.loc[df_inspec["Anomalia 1"] == "NA - NO APLICA", "anomalo"] = np.nan

    df_inspec.sort_values(by=["producto", "fecha_visita"], inplace=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_inspec.to_csv(output_path, index=False)
    print(f"info_inspecciones guardado en: {output_path}  ({len(df_inspec)} filas)")

    return df_inspec


# ---------------------------------------------------------------------------
# Combinación con info_nuevos_anomalos
# ---------------------------------------------------------------------------
def build_inspecciones_etiquetadas(
    inspecciones: pd.DataFrame,
    nuevas_inspecciones: pd.DataFrame,
    output_path: str | Path = None,
) -> pd.DataFrame:
    """
    Combina info_inspecciones con info_nuevos_anomalos en una única base.

    Lógica:
      - Outer join entre ambos sobre 'producto'.
      - Se eliminan filas sin etiqueta en ninguna de las dos fuentes.
      - anomalo final = max(old_anomalo, nuevo_anomalo).
      - Se eliminan filas donde anomalo final es NaN.

    Parameters
    ----------
    inspecciones        : df de build_info_inspecciones (tiene columna 'anomalo').
    nuevas_inspecciones : df de build_info_nuevos_anomalos (tiene 'PRODUCTO' y 'anomalo').
    output_path         : ruta de salida. Por defecto DATA_PROC/inspecciones_etiquetadas.csv.

    Returns
    -------
    DataFrame combinado con columnas: producto, old_anomalo, nuevo_anomalo, anomalo.
    """
    if output_path is None:
        output_path = DATA_PROC / "inspecciones_etiquetadas.csv"
    output_path = Path(output_path)

    # Preparar nuevas inspecciones
    nuevas = nuevas_inspecciones.copy()
    nuevas = nuevas[~nuevas["PRODUCTO"].isna()]
    nuevas.rename(columns={
        "anomalo": "nuevo_anomalo", 
        "PRODUCTO": "producto",
        "fecha_visita": "fecha_nueva", 
        "Anomalia 1": "anomalia_nueva"
    }, inplace=True)

    # Merge outer (incluye las fechas)
    inspecciones_fin = inspecciones.merge(
        nuevas[["producto", "nuevo_anomalo", "fecha_nueva", "anomalia_nueva"]],
        on="producto",
        how="outer",
    )

    # Eliminar filas sin etiqueta en ninguna fuente
    inspecciones_fin = inspecciones_fin[
        ~((inspecciones_fin["nuevo_anomalo"].isna()) & (inspecciones_fin["anomalo"].isna()))
    ]

    inspecciones_fin.rename(columns={
        "anomalo": "old_anomalo",
        "fecha_visita": "fecha_old",
        "Anomalia 1": "anomalia_old"
    }, inplace=True)

    # anomalo final = máximo entre ambas fuentes
    inspecciones_fin["anomalo"] = inspecciones_fin[["old_anomalo", "nuevo_anomalo"]].max(axis=1)

    # fecha_visita = fecha correspondiente al anomalo que se usó
    inspecciones_fin["fecha_visita"] = inspecciones_fin.apply(
        lambda row: row["fecha_nueva"] if row["anomalo"] == row["nuevo_anomalo"] else row["fecha_old"],
        axis=1
    )

    inspecciones_fin["anomalia"] = inspecciones_fin.apply(
        lambda row: row["anomalia_nueva"] if row["anomalo"] == row["nuevo_anomalo"] else row["anomalia_old"],
        axis=1
    )

    # Opcional: limpia columnas temporales
    inspecciones_fin.drop(columns=["fecha_old", "fecha_nueva", "anomalia_old", 'anomalia_nueva'], inplace=True)
    

    # Eliminar filas sin etiqueta final
    inspecciones_fin.dropna(subset="anomalo", inplace=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    inspecciones_fin.to_csv(output_path, index=False)
    print()
    print(f"inspecciones_etiquetadas guardado en: {output_path}  ({len(inspecciones_fin)} filas)")

    return inspecciones_fin


# ---------------------------------------------------------------------------
# Función integradora
# ---------------------------------------------------------------------------
def build_all_inspecciones(
    path_inspecciones: str | Path = None,
    path_nuevos: str | Path = None,
    output_dir: str | Path = None,
) -> pd.DataFrame:
    """
    Pipeline completo:
      1. build_info_inspecciones
      2. Carga info_nuevos_anomalos (o lo construye desde el xlsx)
      3. build_inspecciones_etiquetadas

    Parameters
    ----------
    path_inspecciones : ruta al Excel INSPECCIONES_REALIZADAS.xlsx.
    path_nuevos       : ruta al CSV info_nuevos_anomalos.csv
                        (si no existe, lo lee desde el Excel por defecto).
    output_dir        : directorio de salida. Por defecto DATA_PROC.

    Returns
    -------
    DataFrame combinado final.
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        out_inspec = output_dir / "info_inspecciones.csv"
        out_etiq = output_dir / "inspecciones_etiquetadas.csv"
    else:
        out_inspec = None
        out_etiq = None

    df_inspec = build_info_inspecciones(
        path_xlsx=path_inspecciones,
        output_path=out_inspec,
    )

    # Cargar info_nuevos_anomalos desde CSV si existe, si no desde el xlsx
    if path_nuevos is not None:
        df_nuevos = pd.read_csv(path_nuevos)
    else:
        csv_nuevos = DATA_PROC / "info_nuevos_anomalos.csv"
        if csv_nuevos.exists():
            df_nuevos = pd.read_csv(csv_nuevos)
        else:
            df_nuevos = build_info_nuevos_anomalos()

    df_final = build_inspecciones_etiquetadas(
        inspecciones=df_inspec,
        nuevas_inspecciones=df_nuevos,
        output_path=out_etiq,
    )

    consumo = pd.read_parquet(DATA_PROC / 'consumo_011223_01062025_filtrado.parquet', engine="pyarrow")
    df_final = df_final[df_final.producto.isin(consumo.producto)]
    df_final = df_final[['producto', 'fecha_visita', 'anomalo', 'anomalia']]
    df_final.to_csv(DATA_PROC / "info_inspecciones.csv", index = None)
    return df_final


# ---------------------------------------------------------------------------
# Ejecución directa
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df_final = build_all_inspecciones()
    print("\nResumen base final:")
    print(f"  Total filas : {len(df_final)}")
    print(df_final["anomalo"].value_counts(dropna=False))
