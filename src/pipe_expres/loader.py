import os
import pandas as pd

def loader(
    excel_path: str,
    pkl_path: str,
    columnas_renombrar: dict = None,
    id_vars: list = None
) -> pd.DataFrame:
    """
    Carga un DataFrame desde un archivo pickle si existe. Si no, procesa el archivo Excel,
    lo transforma a formato long y guarda el pickle para uso futuro.

    Parámetros
    ----------
    excel_path : str
        Ruta al archivo Excel original.
    pkl_path : str
        Ruta donde se busca/guarda el archivo pickle.
    columnas_renombrar : dict, opcional
        Diccionario para renombrar columnas específicas antes de hacer melt.
    id_vars : list, opcional
        Lista de columnas que serán variables de identificación en el melt.

    Retorna
    -------
    pd.DataFrame
        DataFrame transformado en formato long.
    """
    if os.path.exists(pkl_path):
        print(f"Cargando desde Pickle: {pkl_path}")
        df_long = pd.read_pickle(pkl_path)
    else:
        print(f"Procesando desde Excel y guardando como Pickle: {excel_path} -> {pkl_path}")
        df = pd.read_excel(excel_path)
        # Renombra columnas si corresponde
        if columnas_renombrar is not None:
            df = df.rename(columns=columnas_renombrar)
        # Si no se especifican, usar todas las columnas que no sean fechas
        if id_vars is None:
            id_vars = [
                "CLIENTE_ID", "DIRECCION", "TRANSFORMADOR_ID", "UBI_TRANSFORMADOR",
                "CELDA", "MUNICIPIO", "BARRIO", "ESTADO_CLIENTE", "ESTRATO",
                "D_CLASE_SERVICIO_MES", "ULTIMO CAMBIO DE MEDIDOR",
                "CLIENTE_ID_TRAIN", "CLIENTE_ID_TEST"
            ]
        # Melt
        df_long = df.melt(
            id_vars=id_vars,
            var_name="fecha",
            value_name="consumo"
        )
        # Ajuste de fechas
        df_long['fecha'] = pd.to_datetime(df_long['fecha'], dayfirst=True, errors='coerce')
        # Guardar como Pickle
        df_long.to_pickle(pkl_path)
        print(f"Pickle guardado en: {pkl_path}")
    return df_long