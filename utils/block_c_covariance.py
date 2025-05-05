# utils/block_c_covariance.py

import numpy as np
import pandas as pd
import warnings
from arch import arch_model
from sklearn.covariance import LedoitWolf
from tqdm import tqdm

# --- GARCH volatility estimator ---
def compute_garch_volatility(series, p=1, q=1):
    try:
        model = arch_model(series.dropna(), vol='Garch', p=p, q=q)
        result = model.fit(disp='off')
        return result.conditional_volatility
    except Exception as e:
        warnings.warn(f"GARCH failed for {series.name}: {e}")
        return pd.Series(index=series.index, data=np.nan)

# --- Shrinked Covariance Estimator ---
def compute_shrunk_cov_matrix(tickers_subset, returns_df, weight_garch=0.6):
    returns = returns_df[tickers_subset].dropna()

    if returns.shape[0] < 30:
        raise ValueError(f"Not enough data for {tickers_subset}")

    # --- GARCH volatility matrix ---
    garch_vols = pd.DataFrame(index=returns.index, columns=tickers_subset)
    for ticker in tickers_subset:
        garch_vols[ticker] = compute_garch_volatility(returns[ticker])
    garch_vols = garch_vols.ffill().bfill()

    # --- Chuẩn hóa độ lệch chuẩn ---
    std_vector = garch_vols.iloc[-1] / 100  # chuyển từ % về decimal

    if std_vector.isnull().any():
        raise ValueError(f"NaN in GARCH volatility for {tickers_subset}")

    # --- Tính covariance theo GARCH ---
    corr_matrix = returns.corr().values
    D = np.diag(std_vector.values)
    cov_garch = D @ corr_matrix @ D

    # --- Ledoit-Wolf Shrinkage ---
    lw = LedoitWolf()
    cov_lw = lw.fit(returns).covariance_

    # --- Kết hợp GARCH + Ledoit-Wolf ---
    cov_combined = weight_garch * cov_garch + (1 - weight_garch) * cov_lw

    # --- Kiểm tra PSD ---
    eigvals = np.linalg.eigvalsh(cov_combined)
    if np.any(eigvals <= 0):
        warnings.warn(f"Cov matrix not PSD for {tickers_subset}. Fallback to Ledoit-Wolf.")
        return pd.DataFrame(cov_lw, index=tickers_subset, columns=tickers_subset)

    return pd.DataFrame(cov_combined, index=tickers_subset, columns=tickers_subset)

# --- Main block ---
def run(selected_combinations, returns_pivot_stocks):
    cov_matrix_dict = {}
    print("🔁 Calculating Shrunk Covariance Matrices...")

    for combo in tqdm(selected_combinations, desc="Portfolio Combinations"):
        tickers_subset = combo.split('-')
        try:
            cov_matrix = compute_shrunk_cov_matrix(tickers_subset, returns_pivot_stocks)
            cov_matrix_dict[combo] = cov_matrix
        except Exception as e:
            warnings.warn(f"[ERROR] {combo}: {e}")
            continue

    print(f"✅ Done. Total valid portfolios: {len(cov_matrix_dict)}")
    return cov_matrix_dict
