import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from scipy.stats import ttest_ind, mannwhitneyu
from tqdm import tqdm

tqdm.pandas()

def _process_group(group, n_regimes, test, alpha):
    series = group.set_index('fecha')['consumo']
    if len(series) < 2:
        return pd.Series({
            'regime_means': np.nan,
            'p_value': np.nan,
            'anomaly': np.nan,
            'regimes': np.nan
        })
    try:
        model = MarkovRegression(series, k_regimes=n_regimes, trend='t', switching_variance=True)
        res = model.fit(disp=False)
        regimes = res.smoothed_marginal_probabilities.idxmax(axis=1).values
        regime_vals = [series.values[regimes == i] for i in range(n_regimes)]
        regime_means = [np.mean(vals) for vals in regime_vals]
        if test == 'ttest':
            stat, p_value = ttest_ind(regime_vals[0], regime_vals[1], equal_var=False, nan_policy='omit')
        elif test == 'mannwhitney':
            stat, p_value = mannwhitneyu(regime_vals[0], regime_vals[1], alternative='two-sided')
        else:
            raise ValueError("test must be 'ttest' or 'mannwhitney'")
        anomaly = p_value < alpha
        return pd.Series({
            'regime_means': regime_means,
            'p_value': p_value,
            'anomaly': anomaly,
            'regimes': regimes
        })
    except Exception:
        # print(f'SDV NO CONVERGE para CLIENTE_ID {group["CLIENTE_ID"].iloc[0]}')
        return pd.Series({
            'regime_means': np.nan,
            'p_value': np.nan,
            'anomaly': np.nan,
            'regimes': np.nan
        })

def detect_markov_regimes_anomaly(df, n_regimes=2, id_col='CLIENTE_ID', test='ttest', alpha=0.05):
    """
    Identifies regimes in a time series using Markov regression and tests for anomalies between regimes,
    for each CLIENTE_ID in the dataframe.
    """
    results = df.groupby(id_col).progress_apply(
        _process_group, n_regimes=n_regimes, test=test, alpha=alpha
    ).reset_index()
    return results