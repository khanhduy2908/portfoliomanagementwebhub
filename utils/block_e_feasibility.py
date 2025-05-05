### BLOCK E: Feasibility Check of Portfolios (Professional Implementation)

```python
import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError

# --- Initialize Valid Portfolio List ---
valid_combinations = []
invalid_log = []

# --- Run Feasibility Checks ---
for combo in adj_returns_combinations.keys():
    tickers = combo.split('-')

    try:
        # Retrieve expected return vector
        mu = np.array([adj_returns_combinations[combo][t] for t in tickers]) / 100

        # Retrieve covariance matrix
        cov_df = cov_matrix_dict.get(combo)
        if cov_df is None:
            raise ValueError("Missing covariance matrix")

        cov = cov_df.loc[tickers, tickers].values

        # Check for NaN, Inf, and negative eigenvalues
        if np.any(np.isnan(mu)) or np.any(np.isinf(mu)):
            raise ValueError("Expected returns contain NaN or Inf")

        if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
            raise ValueError("Covariance matrix contains NaN or Inf")

        if np.all(mu <= 0):
            raise ValueError("All expected returns are non-positive")

        eigvals = np.linalg.eigvalsh(cov)
        if np.any(eigvals < -1e-6):
            raise ValueError("Covariance matrix is not PSD")

        # If all checks pass, mark as valid
        valid_combinations.append(combo)

    except Exception as e:
        invalid_log.append((combo, str(e)))
        continue

# --- Summary Report ---
print("\nPortfolio Feasibility Summary")
print("-------------------------------------")
print(f"✅ Valid Portfolios: {len(valid_combinations)}")
print(f"❌ Invalid Portfolios: {len(invalid_log)}")

if invalid_log:
    print("\nDetails of Invalid Portfolios:")
    for combo, reason in invalid_log:
        print(f" - {combo}: {reason}")
```
