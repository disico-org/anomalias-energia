import pandas as pd
import numpy as np
from statsmodels.api import OLS, add_constant
from statsmodels.stats.diagnostic import breaks_cusumolsresid
from tqdm.auto import tqdm

def detectar_cambio_estructural_cusum(df, id_col='CLIENTE_ID', fecha_col='fecha', consumo_col='consumo', alpha=0.01):
    """
    Aplica test CUSUM a cada cliente para detectar cambio estructural en su serie de consumo.

    Parámetros:
    - df: DataFrame con columnas de cliente, fecha y consumo.
    - id_col: nombre de la columna con ID de cliente.
    - fecha_col: nombre de la columna de fechas.
    - consumo_col: nombre de la columna de consumo.
    - alpha: nivel de significancia (0.01 para 99%).

    Devuelve:
    - DataFrame con resultados: p-value y etiqueta de cambio.
    """
    resultados = []

    grupos = list(df.groupby(id_col))
    for cliente_id, grupo in tqdm(grupos, desc="Detectando cambio estructural"):
        grupo_ordenado = grupo.sort_values(fecha_col)
        y = grupo_ordenado[consumo_col].values

        if len(y) < 10 or np.isnan(y).any():
            continue  # omitimos series muy cortas o con nulos

        X = np.arange(len(y))
        X_const = add_constant(X)

        try:
            modelo = OLS(y, X_const).fit()
            _, pval, _ = breaks_cusumolsresid(modelo.resid)
            cambio = pval < alpha
            resultados.append({
                id_col: cliente_id,
                'cusum_pval': pval,
                'cambio_estructural': cambio
            })
        except Exception as e:
            resultados.append({
                id_col: cliente_id,
                'cusum_pval': np.nan,
                'cambio_estructural': None,
                'error': str(e)
            })

    return pd.DataFrame(resultados)
