import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_no_validas_loop(df, id_col='CLIENTE_ID', fecha_col='fecha', valor_col='consumo', umbral_nulos=0.2):
    """
    Identifica clientes cuya serie temporal tiene:
    - Más del `umbral_nulos` de valores nulos.
    - Varianza cero en la serie (serie plana).

    Parámetros:
    - df: DataFrame con columnas cliente, fecha y consumo.
    - id_col: nombre de la columna de ID de cliente.
    - fecha_col: nombre de la columna de fecha.
    - valor_col: nombre de la columna de valores a evaluar (ej. consumo).
    - umbral_nulos: proporción máxima tolerada de valores nulos (por defecto 0.2 = 20%).

    Retorna:
    - DataFrame con IDs y motivos de exclusión.
    """

    resultados = []

    for id in df[id_col].unique():
        total = len(df[df[id_col] == id].consumo)
        proporcion_nulos = df[df[id_col] == id][valor_col].isna().sum() / total if total > 0 else 1
        varianza = df[df[id_col] == id].consumo.var()
        

        motivo = None
        if proporcion_nulos > umbral_nulos:
            motivo = 'Demasiados nulos'
        
        elif varianza < 1e-6:
            motivo = 'Serie sin varianza'

        if motivo:
            resultados.append({
                id_col: id,
                'proporcion_nulos': proporcion_nulos,
                'varianza': varianza,
                'motivo': motivo
            })

    return pd.DataFrame(resultados)

def get_no_validas(df, cliente_col="CLIENTE_ID", consumo_col="consumo", min_nulos=0.4):
    # Calcula varianza y proporción de nulos por cliente
    resumen = df.groupby(cliente_col)[consumo_col].agg([
        ("varianza", "var"),
        ("n_nulos", lambda x: x.isna().sum()),
        ("total", "count")
    ]).reset_index()
    resumen["prop_nulos"] = resumen["n_nulos"] / resumen["total"]

    # Series sin varianza
    sin_varianza = resumen[resumen["varianza"] < 1e-6][cliente_col].tolist()
    # Series con demasiados nulos
    demasiados_nulos = resumen[resumen["prop_nulos"] > min_nulos][cliente_col].tolist()

    # Construye DataFrame de salida
    resultados = []
    for cid in sin_varianza:
        resultados.append({"CLIENTE_ID": cid, "motivo": "Serie sin varianza"})
    for cid in demasiados_nulos:
        resultados.append({"CLIENTE_ID": cid, "motivo": "Demasiados nulos"})
    return pd.DataFrame(resultados)



def plot_series_no_validas(df_long, id_var, id_nulo, cliente_col="CLIENTE_ID"):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    df_long[df_long[cliente_col]==id_var].plot(x='fecha', y='consumo', ax=axs[0], title='Serie sin varianza ID_CLIENTE_ID: ' + str(id_var))
    df_long[df_long[cliente_col]==id_var].plot(x='fecha', y='consumo', ax=axs[1], title='Serie con demasiados nulos ID_CLIENTE_ID: ' + str(id_nulo))
    axs[0].set_xlabel('Fecha')
    axs[0].set_ylabel('Consumo')
    plt.suptitle('Ejemplo de series no válidas detectadas automáticamente', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()