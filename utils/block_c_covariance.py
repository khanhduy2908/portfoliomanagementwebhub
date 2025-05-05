import warnings
import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.covariance import LedoitWolf
from tqdm import tqdm

# --- Helper: GARCH volatility ---
def compute_garch_volatility(series, p=1, q=1):
    try:
        model = arch_model(series.dropna(), vol='Garch', p=p, q=q)
        result = model.fit(disp='off')
        return result.conditional_volatility
    except Exception as e:
        warnings.warn(f"GARCH failed for {series.name}: {e}")
        return pd.Series(index=series.index, data=np.nan)

# --- Main function ---
def run(selected_combinations, returns_pivot_stocks, weight_garch=0.6):
    cov_matrix_dict = {}
    print("Calculating Shrunk Covariance Matrices...")

    for combo in tqdm(selected_combinations, desc="Portfolio Combinations"):
        tickers_subset = combo.split('-')
        try:
            returns = returns_pivot_stocks[tickers_subset].dropna()

            if returns.shape[0] < 30:
                warnings.warn(f"Not enough data for {tickers_subset}")
                continue

            garch_vols = pd.DataFrame(index=returns.index, columns=tickers_subset)
            for ticker in tickers_subset:
                garch_vols[ticker] = compute_garch_volatility(returns[ticker])

            garch_vols = garch_vols.ffill().bfill()
            std_vector = garch_vols.iloc[-1] / 100

            if std_vector.isnull().any():
                warnings.warn(f"NaN in GARCH volatility for {tickers_subset}")
                continue

            corr_matrix = returns.corr().values
            D = np.diag(std_vector.values)
            cov_garch = D @ corr_matrix @ D

            lw = LedoitWolf()
            cov_lw = lw.fit(returns).covariance_

            cov_combined = weight_garch * cov_garch + (1 - weight_garch) * cov_lw

            eigvals = np.linalg.eigvalsh(cov_combined)
            if np.any(eigvals <= 0):
                warnings.warn(f"Covariance matrix not PSD for {tickers_subset}. Using fallback = Ledoit-Wolf.")
                cov_matrix = pd.DataFrame(cov_lw, index=tickers_subset, columns=tickers_subset)
            else:
                cov_matrix = pd.DataFrame(cov_combined, index=tickers_subset, columns=tickers_subset)

            cov_matrix_dict[combo] = cov_matrix

        except Exception as e:
            warnings.warn(f"[ERROR] {combo}: {e}")
            continue

    print(f"Done. Total valid portfolios: {len(cov_matrix_dict)}")
    return cov_matrix_dict
