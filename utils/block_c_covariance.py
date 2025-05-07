import numpy as np
import pandas as pd
import warnings
from arch import arch_model
from sklearn.covariance import LedoitWolf
import config

# --- A. Tính biến động theo GARCH ---
def compute_garch_volatility(series, p=1, q=1):
    try:
        model = arch_model(series.dropna(), vol='Garch', p=p, q=q)
        result = model.fit(disp='off')
        return result.conditional_volatility
    except Exception as e:
        warnings.warn(f"[GARCH] {series.name} failed: {e}")
        return pd.Series(index=series.index, data=np.nan)

# --- B. Tính ma trận hiệp phương sai kết hợp ---
def compute_shrunk_cov_matrix(tickers, returns_df, weight_garch=None):
    if weight_garch is None:
        weight_garch = getattr(config, 'weight_garch', 0.6)

    returns = returns_df[list(tickers)].dropna()
    if returns.shape[0] < 30:
        raise ValueError(f"[DATA] Insufficient return data for {tickers}")

    # Step 1: Tính volatility từng mã bằng GARCH
    garch_vols = pd.DataFrame(index=returns.index, columns=tickers)
    for ticker in tickers:
        garch_vols[ticker] = compute_garch_volatility(returns[ticker])
    garch_vols = garch_vols.ffill().bfill()

    std_vector = garch_vols.iloc[-1] / 100
    if std_vector.isnull().any():
        raise ValueError(f"[VOL] GARCH returned NaNs for {tickers}")

    # Step 2: Tính ma trận tương quan
    corr_matrix = returns.corr().values
    D = np.diag(std_vector.values)
    cov_garch = D @ corr_matrix @ D

    # Step 3: Tính Ledoit-Wolf
    try:
        lw = LedoitWolf()
        cov_lw = lw.fit(returns).covariance_
    except Exception as e:
        raise ValueError(f"[Ledoit-Wolf] Failed: {e}")

    # Step 4: Kết hợp theo trọng số
    cov_combined = weight_garch * cov_garch + (1 - weight_garch) * cov_lw

    # Step 5: Kiểm tra ma trận dương xác định
    if np.any(np.linalg.eigvalsh(cov_combined) <= 0):
        warnings.warn(f"[PSD] Fallback to Ledoit-Wolf for {tickers}")
        return pd.DataFrame(cov_lw, index=tickers, columns=tickers)

    return pd.DataFrame(cov_combined, index=tickers, columns=tickers)

# --- C. Hàm chính ---
def run(selected_combinations, returns_pivot_stocks):
    cov_matrix_dict = {}
    failed = []

    for combo in selected_combinations:
        try:
            cov_matrix = compute_shrunk_cov_matrix(combo, returns_pivot_stocks)
            cov_matrix_dict[combo] = cov_matrix
        except Exception as e:
            warnings.warn(f"[BLOCK C] {combo} failed: {e}")
            failed.append(combo)

    if not cov_matrix_dict:
        raise ValueError("❌ No valid covariance matrices computed.")

    config.failed_cov_combinations = failed
    return cov_matrix_dict
