# utils/block_h_complete_portfolio.py

import numpy as np
from scipy.optimize import minimize_scalar

def run(hrp_result_dict, adj_returns_combinations, cov_matrix_dict,
        rf, A, total_capital, risk_score, y_min=0.6, y_max=0.9):

    if not hrp_result_dict:
        raise ValueError("No valid HRP-CVaR results from Block G.")

    # 1. Select best portfolio by Sharpe Ratio
    best_key = max(hrp_result_dict, key=lambda k: hrp_result_dict[k]['Sharpe Ratio'])
    best_portfolio = hrp_result_dict[best_key]
    tickers = list(best_portfolio['Weights'].keys())
    weights = np.array([best_portfolio['Weights'][t] for t in tickers])
    weights /= weights.sum()

    # 2. Compute expected return and volatility
    portfolio_name = '-'.join(best_key) if isinstance(best_key, tuple) else str(best_key)
    mu = np.array([adj_returns_combinations[best_key][t] for t in tickers]) / 100
    cov = cov_matrix_dict[best_key].loc[tickers, tickers].values
    sigma_p = np.sqrt(weights.T @ cov @ weights)
    mu_p = weights @ mu

    if mu_p <= 0 or sigma_p <= 0:
        raise ValueError("Selected portfolio has non-positive return or volatility.")

    # 3. Determine max risk-free ratio based on risk score
    if 10 <= risk_score <= 17:
        max_rf_ratio = 0.40
    elif 18 <= risk_score <= 27:
        max_rf_ratio = 0.20
    elif 28 <= risk_score <= 40:
        max_rf_ratio = 0.10
    else:
        raise ValueError("Risk score must be between 10 and 40.")

    # Enforce y bounds: max_rf_ratio => y ≤ 1 - max_rf_ratio
    adjusted_y_min = max(y_min, 1 - max_rf_ratio)
    adjusted_y_max = min(y_max, 1 - max_rf_ratio)

    if adjusted_y_max <= adjusted_y_min:
        raise ValueError(f"Risk constraints too tight: upper_bound={adjusted_y_max:.2f} <= lower_bound={adjusted_y_min:.2f}")

    # 4. Optimize utility
    def neg_utility(y):
        expected_rc = y * mu_p + (1 - y) * rf
        sigma_c = y * sigma_p
        return -(expected_rc - 0.5 * A * sigma_c**2)

    result = minimize_scalar(neg_utility, bounds=(adjusted_y_min, adjusted_y_max), method='bounded')
    y_opt = result.x
    y_capped = y_opt  # Already bounded

    # 5. Final metrics
    expected_rc = y_capped * mu_p + (1 - y_capped) * rf
    sigma_c = y_capped * sigma_p
    utility = expected_rc - 0.5 * A * sigma_c**2

    capital_risky = y_capped * total_capital
    capital_rf = total_capital - capital_risky
    capital_alloc = {t: capital_risky * w for t, w in zip(tickers, weights)}

    # 6. Output dictionary
    portfolio_info = {
        'portfolio_name': portfolio_name,
        'mu': mu_p,
        'sigma': sigma_p,
        'rf': rf,
        'y_opt': y_opt,
        'y_capped': y_capped,
        'A': A,
        'expected_rc': expected_rc,
        'sigma_c': sigma_c,
        'utility': utility,
        'capital_risky': capital_risky,
        'capital_rf': capital_rf,
        'risk_score': risk_score,
        'max_rf_ratio': max_rf_ratio,
        'y_min_enforced': adjusted_y_min,
        'y_max_enforced': adjusted_y_max
    }

    return (
        best_portfolio, y_capped, capital_alloc,
        sigma_c, expected_rc, weights, tickers,
        portfolio_info, sigma_p, mu, y_opt, mu_p, cov
    )
