# utils/block_c_covariance.py

import numpy as np
import pandas as pd
import warnings
from arch import arch_model
from sklearn.covariance import LedoitWolf
import config

def compute_garch_volatility(series, p=1, q=1):
    try:
        model = arch_model(series.dropna(), vol='Garch', p=p, q=q)
        result = model.fit(disp='off')
        return result.conditional_volatility
    except Exception as e:
        warnings.warn(f"GARCH failed for {series.name}: {e}")
        return pd.Series(index=series.index, data=np.nan)

def compute_shrunk_cov_matrix(tickers_subset, returns_df, weight_garch=None):
    if weight_garch is None:
        weight_garch = config.weight_garch if hasattr(config, 'weight_garch') else 0.6

    returns = returns_df[list(tickers_subset)].dropna()

    if returns.shape[0] < 30:
        raise ValueError(f"Not enough data for {tickers_subset}")

    garch_vols = pd.DataFrame(index=returns.index, columns=tickers_subset)
    for ticker in tickers_subset:
        garch_vols[ticker] = compute_garch_volatility(returns[ticker])
    garch_vols = garch_vols.ffill().bfill()

    std_vector = garch_vols.iloc[-1] / 100  # from % to decimal

    if std_vector.isnull().any():
        raise ValueError(f"NaN in GARCH volatility for {tickers_subset}")

    corr_matrix = returns.corr().values
    D = np.diag(std_vector.values)
    cov_garch = D @ corr_matrix @ D

    try:
        lw = LedoitWolf()
        cov_lw = lw.fit(returns).covariance_
    except Exception as e:
        warnings.warn(f"Ledoit-Wolf failed: {e}")
        raise

    cov_combined = weight_garch * cov_garch + (1 - weight_garch) * cov_lw

    eigvals = np.linalg.eigvalsh(cov_combined)
    if np.any(eigvals <= 0):
        warnings.warn(f"Cov matrix not PSD for {tickers_subset}. Fallback to Ledoit-Wolf.")
        return pd.DataFrame(cov_lw, index=tickers_subset, columns=tickers_subset)

    return pd.DataFrame(cov_combined, index=tickers_subset, columns=tickers_subset)

def run(selected_combinations, returns_pivot_stocks):
    cov_matrix_dict = {}
    failed_combinations = []

    for combo in selected_combinations:
        try:
            cov_matrix = compute_shrunk_cov_matrix(combo, returns_pivot_stocks)
            cov_matrix_dict[combo] = cov_matrix
        except Exception as e:
            warnings.warn(f"[ERROR] {combo}: {e}")
            failed_combinations.append(combo)
            continue

    if len(cov_matrix_dict) == 0:
        raise ValueError("❌ Không tính được ma trận hiệp phương sai nào.")

    return cov_matrix_dict
