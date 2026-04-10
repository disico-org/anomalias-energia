"""
build_anomalos.py

Genera los archivos CSV:
  - info_anomalos.csv        (fuente: reporte de irregularidades histórico xlsx)
  - info_nuevos_anomalos.csv (fuente: reporte de irregularidades nuevo xlsx)

Basado en los notebooks:
  - notebooks/EMCALI/construccion_base.ipynb
  - notebooks/EMCALI/rev_inspec.ipynb
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Rutas por defecto (relativas a la raíz del proyecto)
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
# info_anomalos
# ---------------------------------------------------------------------------
def build_info_anomalos(
    path_xlsx: str | Path = None,
    output_path: str | Path = None,
) -> pd.DataFrame:
    """
    Procesa el reporte histórico de irregularidades y genera info_anomalos.csv.

    Fuente esperada: lv_lega_insp_irregularidad_report02102025.xlsx
    Columnas de salida:
        fecha_visita, producto, direccion,
        Anomalia 1, Anomalia 2, Anomalia 3, Anomalia 4,
        Actividad Comercial, anomalo

    Parameters
    ----------
    path_xlsx   : ruta al Excel fuente. Por defecto usa DATA_RAW.
    output_path : ruta donde guardar el CSV. Por defecto usa DATA_PROC.

    Returns
    -------
    DataFrame con los anomalos procesados.
    """
    if path_xlsx is None:
        path_xlsx = DATA_RAW / "lv_lega_insp_irregularidad_report02102025.xlsx"
    if output_path is None:
        output_path = DATA_PROC / "info_anomalos.csv"

    path_xlsx = Path(path_xlsx)
    output_path = Path(output_path)

    df = pd.read_excel(path_xlsx)
    df.drop_duplicates(inplace=True)

    df_anomalos = (
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

    df_anomalos["anomalo"] = 1
    df_anomalos.rename(
        columns={
            "Fecha Visita": "fecha_visita",
            "Producto": "producto",
            "Direccion": "direccion",
        },
        inplace=True,
    )
    df_anomalos["fecha_visita"] = pd.to_datetime(df_anomalos["fecha_visita"], dayfirst=True)
    df_anomalos.sort_values(by=["producto", "fecha_visita"], inplace=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_anomalos.to_csv(output_path, index=False)
    print(f"info_anomalos guardado en: {output_path}  ({len(df_anomalos)} filas)")

    return df_anomalos 


# ---------------------------------------------------------------------------
# info_nuevos_anomalos
# ---------------------------------------------------------------------------
def build_info_nuevos_anomalos(
    path_xlsx: str | Path = None,
    output_path: str | Path = None,
) -> pd.DataFrame:
    """
    Procesa el reporte nuevo de irregularidades y genera info_nuevos_anomalos.csv.

    Fuente esperada: 201225_reporte_irregularidades_2025-17-16 08-55.xlsx
    Columnas de salida:
        PRODUCTO, FECHA EJECUCION, IRREGULARIDAD_1, OBSERVACIÓN IRREGULARIDAD_1

    Reglas:
      - Se excluyen filas con IRREGULARIDAD_1 == '16 - OTRO'.
      - Se conservan solo los registros que no están en NO_ANOMALO
        (son los efectivamente anómalos).

    Parameters
    ----------
    path_xlsx   : ruta al Excel fuente. Por defecto usa DATA_RAW.
    output_path : ruta donde guardar el CSV. Por defecto usa DATA_PROC.

    Returns
    -------
    DataFrame con los nuevos anomalos procesados.
    """
    if path_xlsx is None:
        path_xlsx = (
            DATA_RAW / "201225_reporte_irregularidades_2025-17-16 08-55.xlsx"
        )
    if output_path is None:
        output_path = DATA_PROC / "info_nuevos_anomalos.csv"

    path_xlsx = Path(path_xlsx)
    output_path = Path(output_path)

    df = pd.read_excel(path_xlsx)
    df["fecha_visita"] = pd.to_datetime(df["FECHA EJECUCION"])

    # Excluir "16 - OTRO" (sin información útil)
    df = df[df["IRREGULARIDAD_1"] != "16 - OTRO"].copy()

    # Columna anomalo (para referencia interna; no se guarda en el CSV)
    df["anomalo"] = 0
    df.loc[~df["IRREGULARIDAD_1"].isin(NO_ANOMALO), "anomalo"] = 1
    df['Anomalia 1'] = df["IRREGULARIDAD_1"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(
        f"info_nuevos_anomalos guardado en: {output_path}  ({len(df)} filas)"
    )

    return df


# ---------------------------------------------------------------------------
# Combinación: anomalos_etiquetados
# ---------------------------------------------------------------------------
def build_anomalos_etiquetados(
    inspecciones: pd.DataFrame,
    nuevas_inspecciones: pd.DataFrame,
    output_path: str | Path = None,
) -> pd.DataFrame:
    """
    Combina el reporte histórico y el nuevo en una única base etiquetada.

    Lógica:
      - Outer join entre ambos sobre 'producto'.
      - anomalo final = max(old_anomalo, nuevo_anomalo).
      - Se eliminan filas donde ambas fuentes son NaN o donde anomalo es NaN.

    Parameters
    ----------
    inspecciones       : df de build_info_anomalos (tiene columna 'anomalo').
    nuevas_inspecciones: df de build_info_nuevos_anomalos (tiene 'PRODUCTO' y 'anomalo').
    output_path        : ruta donde guardar el CSV. Por defecto DATA_PROC/anomalos_etiquetados.csv.

    Returns
    -------
    DataFrame combinado con columnas: producto, old_anomalo, nuevo_anomalo, anomalo.
    """
    if output_path is None:
        output_path = DATA_PROC / "anomalos_etiquetados.csv"
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
        "fecha_visita": "fecha_old" ,
        "Anomalia 1": "anomalia_old"
    }, inplace=True)

    # anomalo final = máximo entre ambas fuentes
    inspecciones_fin["anomalo"] = inspecciones_fin[["old_anomalo", "nuevo_anomalo"]].max(axis=1)

    # fecha_visita = fecha correspondiente al anomalo que se usó
    inspecciones_fin["fecha_visita"] = inspecciones_fin.apply(
        lambda row: row["fecha_nueva"] if row["anomalo"] == row["nuevo_anomalo"] else row["fecha_old"],
        axis=1
    )

    inspecciones_fin["explicacion"] = inspecciones_fin.apply(
        lambda row: row["anomalia_nueva"] if row["anomalo"] == row["nuevo_anomalo"] else row["anomalia_old"],
        axis=1
    )

    # Opcional: limpia columnas temporales
    inspecciones_fin.drop(columns=["fecha_old", "fecha_nueva"], inplace=True)

    # Eliminar filas sin etiqueta final
    inspecciones_fin.dropna(subset="anomalo", inplace=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    inspecciones_fin.to_csv(output_path, index=False)
    print(f"anomalos_etiquetados guardado en: {output_path}  ({len(inspecciones_fin)} filas)")

    return inspecciones_fin


# ---------------------------------------------------------------------------
# Función integradora
# ---------------------------------------------------------------------------
def build_all_anomalos(
    path_historico: str | Path = None,
    path_nuevo: str | Path = None,
    output_dir: str | Path = None,
) -> pd.DataFrame:
    """
    Ejecuta el pipeline completo:
      1. build_info_anomalos
      2. build_info_nuevos_anomalos
      3. build_anomalos_etiquetados

    Parameters
    ----------
    path_historico : ruta al Excel histórico de irregularidades.
    path_nuevo     : ruta al Excel nuevo de irregularidades.
    output_dir     : directorio de salida para los CSVs.
                     Por defecto: data/EMCALI/datos_procesados/

    Returns
    -------
    DataFrame combinado final (anomalos_etiquetados).
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        out_anomalos = output_dir / "info_anomalos.csv"
        out_nuevos = output_dir / "info_nuevos_anomalos.csv"
        out_etiquetados = output_dir / "anomalos_etiquetados.csv"
    else:
        out_anomalos = None
        out_nuevos = None
        out_etiquetados = None

    df_anomalos = build_info_anomalos(
        path_xlsx=path_historico,
        output_path=out_anomalos,
    )
    df_nuevos = build_info_nuevos_anomalos(
        path_xlsx=path_nuevo,
        output_path=out_nuevos,
    )
    df_final = build_anomalos_etiquetados(
        inspecciones=df_anomalos,
        nuevas_inspecciones=df_nuevos,
        output_path=out_etiquetados,
    )

    return df_final


# ---------------------------------------------------------------------------
# Ejecución directa
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df_final = build_all_anomalos()
    print("\nResumen base final:")
    print(f"  Total filas : {len(df_final)}")
    print(df_final["anomalo"].value_counts(dropna=False))
