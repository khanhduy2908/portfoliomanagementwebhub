import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
import warnings

def run(adj_returns_combinations, cov_matrix_dict):
    valid_combinations = []
    invalid_log = []

    for combo in adj_returns_combinations:
        tickers = list(combo)

        # --- Step 1: Expected Return Vector ---
        try:
            mu = np.array([adj_returns_combinations[combo][t] for t in tickers]) / 100
        except KeyError:
            invalid_log.append((combo, "Missing expected return values"))
            continue

        # --- Step 2: Covariance Matrix ---
        try:
            cov_df = cov_matrix_dict[combo]
            cov = cov_df.loc[tickers, tickers].values
        except Exception as e:
            invalid_log.append((combo, f"Invalid covariance matrix: {e}"))
            continue

        # --- Step 3: Validation Checks ---
        if np.any(np.isnan(mu)) or np.any(np.isinf(mu)):
            invalid_log.append((combo, "NaN or Inf in expected returns"))
            continue

        if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
            invalid_log.append((combo, "NaN or Inf in covariance matrix"))
            continue

        if np.all(mu <= 0):
            invalid_log.append((combo, "All expected returns are non-positive"))
            continue

        try:
            eigvals = np.linalg.eigvalsh(cov)
            if np.any(eigvals < -1e-6):
                invalid_log.append((combo, "Covariance matrix is not positive semi-definite"))
                continue
        except LinAlgError as e:
            invalid_log.append((combo, f"Eigenvalue decomposition failed: {e}"))
            continue

        # --- Step 4: Append Valid ---
        valid_combinations.append(combo)

    # --- Optional Logging ---
    print("\n📊 Portfolio Feasibility Summary")
    print("--------------------------------------------------")
    print(f"✅ Valid portfolios:   {len(valid_combinations)}")
    print(f"❌ Invalid portfolios: {len(invalid_log)}")

    if invalid_log:
        print("⚠ Reasons:")
        for combo, reason in invalid_log:
            warnings.warn(f"{combo}: {reason}")

    return valid_combinations
