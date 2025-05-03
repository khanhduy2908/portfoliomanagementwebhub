import pandas as pd
import numpy as np
import warnings
from arch import arch_model
from sklearn.covariance import LedoitWolf

# --- GARCH Volatility Estimation ---
def compute_garch_volatility(series, p=1, q=1):
    try:
        model = arch_model(series.dropna(), vol='Garch', p=p, q=q)
        result = model.fit(disp='off')
        return result.conditional_volatility
    except Exception as e:
        warnings.warn(f"[GARCH Failed] {series.name}: {e}")
        return pd.Series(index=series.index, data=np.nan)

# --- Combined Covariance Matrix (GARCH + Ledoit-Wolf Shrinkage) ---
def compute_shrunk_cov_matrix(tickers_subset, returns_df, weight_garch=0.6):
    returns = returns_df[tickers_subset].dropna()

    if returns.shape[0] < 30:
        raise ValueError(f"⚠️ Not enough data for combination: {tickers_subset}")

    # Step 1: GARCH Volatility for each stock
    garch_vols = pd.DataFrame(index=returns.index, columns=tickers_subset)
    for ticker in tickers_subset:
        garch_vols[ticker] = compute_garch_volatility(returns[ticker])

    garch_vols = garch_vols.ffill().bfill()
    std_vector = garch_vols.iloc[-1] / 100

    if std_vector.isnull().any():
        raise ValueError(f"❌ NaN in GARCH volatility for {tickers_subset}")

    # Step 2: Correlation + GARCH-based Covariance
    corr_matrix = returns.corr().values
    D = np.diag(std_vector.values)
    cov_garch = D @ corr_matrix @ D

    # Step 3: Ledoit-Wolf shrinkage estimator
    lw = LedoitWolf()
    cov_lw = lw.fit(returns).covariance_

    # Step 4: Combine both with weight
    cov_combined = weight_garch * cov_garch + (1 - weight_garch) * cov_lw

    # Step 5: Check PSD (Positive Semi-Definite)
    eigvals = np.linalg.eigvalsh(cov_combined)
    if np.any(eigvals <= 0):
        warnings.warn(f"⚠️ Covariance not PSD for {tickers_subset}. Fallback to Ledoit-Wolf.")
        return pd.DataFrame(cov_lw, index=tickers_subset, columns=tickers_subset)

    return pd.DataFrame(cov_combined, index=tickers_subset, columns=tickers_subset)

# --- Main Function: Batch Covariance Estimation ---
def compute_cov_matrices(selected_combinations, returns_pivot_stocks):
    cov_matrix_dict = {}
    for combo in selected_combinations:
        tickers_subset = list(combo)
        combo_label = "-".join(tickers_subset)
        try:
            cov_matrix = compute_shrunk_cov_matrix(tickers_subset, returns_pivot_stocks)
            cov_matrix_dict[combo_label] = cov_matrix
        except Exception as e:
            warnings.warn(f"[ERROR] {combo_label}: {e}")
            continue
    return cov_matrix_dict
