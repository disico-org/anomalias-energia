import numpy as np
import pandas as pd
from tqdm import tqdm

def feature_consumo_log(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['consumo_log'] = np.log(df['consumo'])
    return df

def feature_consumo_prev_year(df: pd.DataFrame, n_months = 4) -> pd.DataFrame:
    print(n_months)
    df = df.copy()
    df['consumo_prev_year'] = df.groupby('CLIENTE_ID')['consumo'].shift(n_months)
    df['consumo_log_prev_year'] = df.groupby('CLIENTE_ID')['consumo_log'].shift(n_months)
    return df

def feature_year_month(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Crear columna del mes actual (año-mes) para identificar el periodo de detección
    df['year_month'] = df['fecha'].dt.to_period('M').astype(str)
    return df

def crear_features(df: pd.DataFrame, feature_funcs: list) -> pd.DataFrame:
    """
    Aplica una lista de funciones de ingeniería de características sobre un DataFrame.
    """
    for func in feature_funcs:
        df = func(df)
    return df