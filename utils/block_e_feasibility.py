import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
import warnings

def run(adj_returns_combinations, cov_matrix_dict):
    valid_combinations = []
    invalid_log = []

    for combo in adj_returns_combinations.keys():
        tickers = list(combo)  # combo là tuple

        # 1. Expected return vector
        try:
            mu = np.array([adj_returns_combinations[combo][t] for t in tickers]) / 100
        except KeyError:
            invalid_log.append((combo, "❌ Missing expected return"))
            continue

        # 2. Covariance matrix
        try:
            cov_df = cov_matrix_dict[combo]
            cov = cov_df.loc[tickers, tickers].values
        except Exception as e:
            invalid_log.append((combo, f"❌ Missing or invalid covariance matrix: {e}"))
            continue

        # 3. Validation
        if np.any(np.isnan(mu)) or np.any(np.isinf(mu)):
            invalid_log.append((combo, "❌ mu contains NaN or Inf"))
            continue

        if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
            invalid_log.append((combo, "❌ cov contains NaN or Inf"))
            continue

        if np.all(mu <= 0):
            invalid_log.append((combo, "❌ All expected returns <= 0"))
            continue

        try:
            eigvals = np.linalg.eigvalsh(cov)
            if np.any(eigvals < -1e-6):
                invalid_log.append((combo, "❌ Covariance matrix not PSD"))
                continue
        except LinAlgError as e:
            invalid_log.append((combo, f"❌ Covariance eig failed: {e}"))
            continue

        # Passed all checks
        valid_combinations.append(combo)

    # Logging
    print("\n📊 Portfolio Feasibility Check Summary")
    print("--------------------------------------------------")
    print(f"✅ Valid combinations: {len(valid_combinations)}")
    print(f"❌ Invalid combinations: {len(invalid_log)}")
    for combo, reason in invalid_log:
        warnings.warn(f"[{combo}] {reason}")

    return valid_combinations
