import pandas as pd
import numpy as np

def filtrar_clientes(
    df_long: pd.DataFrame,
    fecha_col: str = 'fecha',
    consumo_col: str = 'consumo',
    cliente_col: str = 'CLIENTE_ID',
    meses: int = 24,
    medidor_col: str = 'ULTIMO CAMBIO DE MEDIDOR',
    texto_sin_cambio: str = 'SIN CAMBIO MEDIDOR',
    eliminar_recientes: bool = True,
    eliminar_bajo: bool = True,
    umbral_consumo_min: float = 0,
    eliminar_altos: bool = True,
    percentil_alto: float = 0.95,
    solo_panel_completo: bool = False,
    group_cols = ['MUNICIPIO', 'BARRIO', 'ESTRATO',	'D_CLASE_SERVICIO_MES', 'TRANSFORMADOR_ID']
) -> pd.DataFrame:
    """
    Filtra un DataFrame de consumos de clientes aplicando diferentes criterios de depuración.
    """
    # 1. Ordena y limpia los tipos
    df_long = df_long.copy()
    print('Número de clientes originales:', df_long[cliente_col].nunique())

    # Ajuste formato
    for col in group_cols:
        df_long[col] = df_long[col].astype('category')

    
    df_long[fecha_col] = pd.to_datetime(df_long[fecha_col])
    df_long[consumo_col] = pd.to_numeric(df_long[consumo_col], errors='coerce')
    
    # Interpolación de valores nulos en la columna de consumo
    df_long[consumo_col] = df_long.sort_values('fecha').groupby('CLIENTE_ID')[consumo_col].transform(
        lambda x: x.interpolate(method='spline', order=2, limit_direction='both'))


    # 2. Filtros de calidad **ANTES** del recorte de fechas
    if eliminar_bajo:
        clientes_bajo = df_long[df_long[consumo_col] <= umbral_consumo_min][cliente_col].unique()
        df_long = df_long[~df_long[cliente_col].isin(clientes_bajo)]
        print('Número de clientes después de eliminar bajo consumo:', df_long[cliente_col].nunique())

    if eliminar_recientes and medidor_col in df_long.columns:
        # Reemplaza el texto que indica "sin cambio de medidor" por NaN
        df_long.loc[df_long[medidor_col] == texto_sin_cambio, medidor_col] = np.nan
        # Convierte la columna a datetime (maneja errores como NaT)
        df_long[medidor_col] = pd.to_datetime(df_long[medidor_col], errors='coerce')
        # Mantiene clientes cuyo último cambio de medidor fue antes del primer mes permitido o nunca lo han cambiado
        min_fecha = df_long[fecha_col].max() - pd.DateOffset(months=meses)
        clientes_recientes = df_long[df_long[medidor_col] >= min_fecha][cliente_col].unique()
        df_long = df_long[~df_long[cliente_col].isin(clientes_recientes)]
        print('Número de clientes después de eliminar recientes:', df_long[cliente_col].nunique())


    if eliminar_altos:
        for c in df_long['D_CLASE_SERVICIO_MES'].unique():
            consumo_medio = df_long[df_long.D_CLASE_SERVICIO_MES == c].groupby(cliente_col)[consumo_col].mean()
            umbral_alto = consumo_medio.quantile(percentil_alto)
            clientes_muy_altos = consumo_medio[consumo_medio > umbral_alto].index
            df_long = df_long[~df_long[cliente_col].isin(clientes_muy_altos)]
        print('Número de clientes después de eliminar altos:', df_long[cliente_col].nunique())

    # 3. Ahora recorta a los últimos N meses

    max_fecha = df_long[fecha_col].max()
    min_fecha_permitida = max_fecha - pd.DateOffset(months=meses)
    df_long = df_long[df_long[fecha_col] > min_fecha_permitida].copy()

    # 4. (OPCIONAL) Panel completamente balanceado (solo clientes con N registros)
    if solo_panel_completo:
        clientes_completos = (
            df_long.groupby(cliente_col)[fecha_col]
            .nunique().pipe(lambda s: s[s == meses]).index
        )
        df_long = df_long[df_long[cliente_col].isin(clientes_completos)].copy()

    print('Número de clientes después de limpieza:', df_long[cliente_col].nunique())
    return df_long
