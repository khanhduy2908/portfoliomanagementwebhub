
import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError

def portfolio_feasibility_precheck(adj_returns_combinations, cov_matrix_dict):
    valid_combinations = []
    invalid_log = []

    target_combinations = list(adj_returns_combinations.keys())

    for combo in target_combinations:
        tickers = combo.split('-')

        try:
            mu = np.array([adj_returns_combinations[combo][t] for t in tickers]) / 100
        except KeyError:
            invalid_log.append((combo, "Missing expected return"))
            continue

        try:
            cov_df = cov_matrix_dict[combo].copy()
            cov = cov_df.loc[tickers, tickers].values
        except Exception as e:
            invalid_log.append((combo, f"Missing covariance matrix: {e}"))
            continue

        if np.any(np.isnan(mu)) or np.any(np.isinf(mu)):
            invalid_log.append((combo, "mu contains NaN or Inf"))
            continue

        if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
            invalid_log.append((combo, "cov contains NaN or Inf"))
            continue

        if np.all(mu <= 0):
            invalid_log.append((combo, "All expected returns <= 0"))
            continue

        try:
            eigvals = np.linalg.eigvalsh(cov)
            if np.any(eigvals < -1e-6):
                invalid_log.append((combo, "Covariance matrix not PSD"))
                continue
        except LinAlgError as e:
            invalid_log.append((combo, f"Covariance eig failed: {e}"))
            continue

        valid_combinations.append(combo)

    print("\n\U0001f4c8 Portfolio Feasibility Check Summary")
    print("--------------------------------------------------")
    print(f"\u2705 Valid combinations: {len(valid_combinations)}")
    print(f"\u274C Invalid combinations: {len(invalid_log)}")

    if invalid_log:
        print("\nDetails of invalid combinations:")
        for combo, reason in invalid_log:
            print(f" - {combo}: {reason}")

    return valid_combinations, invalid_log
