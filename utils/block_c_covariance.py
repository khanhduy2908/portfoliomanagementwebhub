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
        warnings.warn(f"[GARCH ERROR] {series.name}: {e}")
        return pd.Series(index=series.index, data=np.nan)

def compute_shrunk_cov_matrix(tickers_subset, returns_df, weight_garch=None):
    if weight_garch is None:
        weight_garch = getattr(config, 'weight_garch', 0.6)

    returns = returns_df[list(tickers_subset)].dropna()
    if returns.shape[0] < 30:
        raise ValueError(f"[DATA ERROR] Not enough data for {tickers_subset}")

    # Step 1: GARCH volatility estimation
    garch_vols = pd.DataFrame(index=returns.index, columns=tickers_subset)
    for ticker in tickers_subset:
        garch_vols[ticker] = compute_garch_volatility(returns[ticker])
    garch_vols = garch_vols.ffill().bfill()
    std_vector = garch_vols.iloc[-1] / 100

    if std_vector.isnull().any():
        raise ValueError(f"[VOL ERROR] NaN in GARCH volatility for {tickers_subset}")

    # Step 2: Correlation matrix
    corr_matrix = returns.corr().values
    D = np.diag(std_vector.values)
    cov_garch = D @ corr_matrix @ D

    # Step 3: Ledoit-Wolf shrinkage
    try:
        lw = LedoitWolf()
        cov_lw = lw.fit(returns).covariance_
    except Exception as e:
        warnings.warn(f"[LW ERROR] Ledoit-Wolf failed: {e}")
        raise

    cov_combined = weight_garch * cov_garch + (1 - weight_garch) * cov_lw

    # Step 4: PSD check
    eigvals = np.linalg.eigvalsh(cov_combined)
    if np.any(eigvals <= 0):
        warnings.warn(f"[PSD ERROR] Matrix not PSD. Fallback to Ledoit-Wolf for {tickers_subset}")
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
            warnings.warn(f"[BLOCK C ERROR] {combo}: {e}")
            failed_combinations.append(combo)
            continue

    if not cov_matrix_dict:
        raise ValueError("❌ No valid covariance matrix could be computed.")

    # Optional: store failed combinations in config for review/debug
    config.failed_cov_combinations = failed_combinations

    return cov_matrix_dict
