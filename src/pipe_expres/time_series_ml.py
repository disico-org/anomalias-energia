import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings('ignore')


def _evaluar_grupo(cliente_id, y, train_len, test_len, umbral_mae):
    """Evalúa anomalía ETS sobre arrays ya extraídos (sin filtrar df)."""
    resultados = {"CLIENTE_ID": cliente_id}

    if len(y) <= train_len or len(y) < train_len + test_len or np.std(y[:train_len]) < 1e-3:
        resultados.update({
            "suficientes_datos": False,
            "motivo": "Corta/plana",
            "mae_ets": np.nan,
            "anomalia_ets": None,
        })
        return resultados

    train, test = y[:train_len], y[train_len:train_len + test_len]

    try:
        use_seasonal = len(train) >= 2 * 12
        model_ets = ExponentialSmoothing(
            train,
            trend="add",
            seasonal="add" if use_seasonal else None,
            seasonal_periods=12 if use_seasonal else None,
            initialization_method="estimated",
        ).fit()
        pred_ets = model_ets.forecast(test_len)
        mae_ets = mean_absolute_error(test, pred_ets)
        resultados["mae_ets"] = mae_ets
        resultados["anomalia_ets"] = mae_ets > umbral_mae * np.mean(np.abs(train))
    except Exception as e:
        resultados["suficientes_datos"] = False
        resultados["mae_ets"] = np.nan
        resultados["anomalia_ets"] = None
        resultados["motivo"] = f"Error ETS: {e}"
        return resultados

    resultados["suficientes_datos"] = True
    if "motivo" not in resultados:
        resultados["motivo"] = ""
    return resultados


def evaluar_anomalia_ets_batch(df, train_len=24, test_len=4, umbral_mae=0.95, n_jobs=-1):
    """
    Evalúa anomalía ETS para todos los clientes en paralelo.

    Parámetros
    ----------
    df        : DataFrame con columnas CLIENTE_ID, fecha, consumo
    train_len : meses de entrenamiento
    test_len  : meses de prueba
    umbral_mae: fracción del consumo medio a superar para marcar anomalía
    n_jobs    : núcleos a usar (-1 = todos)

    Retorna
    -------
    DataFrame con resultados por cliente
    """
    # Pre-agrupar una sola vez — evita filtrar el df completo en cada iteración
    grupos = {
        cid: grp.sort_values("fecha")["consumo"].values
        for cid, grp in df.groupby("CLIENTE_ID")
    }

    resultados = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_evaluar_grupo)(cid, y, train_len, test_len, umbral_mae)
        for cid, y in grupos.items()
    )

    return pd.DataFrame(resultados)


def evaluar_anomalia_ets(df, cliente_id, train_len=24, test_len=4, umbral_mae=0.95):
    """
    Evalúa anomalía en la serie de consumo de un cliente usando Exponential Smoothing (ETS).
    """
    y = df[df["CLIENTE_ID"] == cliente_id].sort_values("fecha")["consumo"].values
    return _evaluar_grupo(cliente_id, y, train_len, test_len, umbral_mae)



def confirmar_anomalia_completa(df, cliente_id, train_len=24, test_len=4, umbral_mae=0.6):
    """
    Confirma anomalía usando auto_arima y Prophet.
    """
    sub_df = df[df["CLIENTE_ID"] == cliente_id].sort_values("fecha")
    y = sub_df["consumo"].values
    fechas = sub_df["fecha"].values

    res = {}
    if len(y) < train_len + test_len or np.std(y[:train_len]) < 1e-3:
        res["mae_arima"] = np.nan
        res["anomalia_arima"] = None
        res["mae_prophet"] = np.nan
        res["anomalia_prophet"] = None
        res["error_arima"] = "Corta/plana"
        res["error_prophet"] = "Corta/plana"
        return res

    train, test = y[:train_len], y[train_len:train_len+test_len]
    train_fechas, test_fechas = fechas[:train_len], fechas[train_len:train_len+test_len]

    # auto_arima
    try:
        model_arima = auto_arima(train, seasonal=True, m=12, suppress_warnings=True, error_action='ignore')
        pred_arima = model_arima.predict(n_periods=test_len)
        mae_arima = mean_absolute_error(test, pred_arima)
        res["mae_arima"] = mae_arima
        res["anomalia_arima"] = mae_arima > umbral_mae * np.mean(np.abs(train))
    except Exception as e:
        res["mae_arima"] = np.nan
        res["anomalia_arima"] = None
        res["error_arima"] = str(e)

    # Prophet
    try:
        prophet_df = pd.DataFrame({"ds": train_fechas, "y": train})
        model_prophet = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
        model_prophet.fit(prophet_df)
        future = pd.DataFrame({"ds": test_fechas})
        forecast = model_prophet.predict(future)
        pred_prophet = forecast["yhat"].values
        mae_prophet = mean_absolute_error(test, pred_prophet)
        res["mae_prophet"] = mae_prophet
        res["anomalia_prophet"] = mae_prophet > umbral_mae * np.mean(np.abs(train))
    except Exception as e:
        res["mae_prophet"] = np.nan
        res["anomalia_prophet"] = None
        res["error_prophet"] = str(e)

    return res


def consenso(row):
    votos = sum([
        int(row.get("anomalia_ets") == True),
        int(row.get("anomalia_arima") == True),
        int(row.get("anomalia_prophet") == True),
    ])
    return votos >= 2