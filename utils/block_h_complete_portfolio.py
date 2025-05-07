# utils/block_h_complete_portfolio.py

import numpy as np
from scipy.optimize import minimize_scalar

def run(hrp_result_dict, adj_returns_combinations, cov_matrix_dict,
        rf, A, total_capital, risk_score, y_min=0.6, y_max=0.9):

    if not hrp_result_dict:
        raise ValueError("No valid HRP-CVaR results from Block G.")

    # 1. Select best portfolio based on Sharpe Ratio
    best_key = max(hrp_result_dict, key=lambda k: hrp_result_dict[k]['Sharpe Ratio'])
    best_portfolio = hrp_result_dict[best_key]
    tickers = list(best_portfolio['Weights'].keys())
    weights = np.array([best_portfolio['Weights'][t] for t in tickers])
    weights /= weights.sum()

    # 2. Return and risk calculations
    portfolio_name = '-'.join(best_key) if isinstance(best_key, tuple) else str(best_key)
    mu = np.array([adj_returns_combinations[best_key][t] for t in tickers]) / 100
    cov = cov_matrix_dict[best_key].loc[tickers, tickers].values

    sigma_p = np.sqrt(weights.T @ cov @ weights)
    mu_p = weights @ mu

    if mu_p <= 0 or sigma_p <= 0:
        raise ValueError("Selected portfolio has non-positive return or volatility.")

    # 3. Risk profile constraints (strictly enforce max risk-free ratio)
    if 10 <= risk_score <= 17:
        max_risk_free_ratio = 0.85  # Conservative investors
    elif 18 <= risk_score <= 27:
        max_risk_free_ratio = 0.50  # Balanced investors
    elif 28 <= risk_score <= 40:
        max_risk_free_ratio = 0.20  # Aggressive investors
    else:
        max_risk_free_ratio = 0.0

    min_y_allowed = max(y_min, 1.0 - max_risk_free_ratio)
    upper_y_bound = min(y_max, 1.0)

    if upper_y_bound <= min_y_allowed:
        raise ValueError(f"Risk constraints too tight: upper_y_bound={upper_y_bound} <= min_y_allowed={min_y_allowed}")

    # 4. Optimize utility over allowed y range
    def neg_utility(y):
        expected_rc = y * mu_p + (1 - y) * rf
        sigma_c = y * sigma_p
        return -(expected_rc - 0.5 * A * sigma_c**2)

    result = minimize_scalar(neg_utility, bounds=(min_y_allowed, upper_y_bound), method='bounded')
    y_opt = result.x
    y_capped = np.clip(y_opt, min_y_allowed, upper_y_bound)

    # 5. Final calculations
    expected_rc = y_capped * mu_p + (1 - y_capped) * rf
    sigma_c = y_capped * sigma_p
    utility = expected_rc - 0.5 * A * sigma_c**2

    capital_risky = y_capped * total_capital
    capital_rf = total_capital - capital_risky
    capital_alloc = {t: capital_risky * w for t, w in zip(tickers, weights)}

    portfolio_info = {
        'portfolio_name': portfolio_name,
        'mu': mu_p,
        'sigma': sigma_p,
        'rf': rf,
        'y_opt': y_opt,
        'y_capped': y_capped,
        'A': A,
        'risk_score': risk_score,
        'expected_rc': expected_rc,
        'sigma_c': sigma_c,
        'utility': utility,
        'capital_risky': capital_risky,
        'capital_rf': capital_rf
    }

    return (
        best_portfolio, y_capped, capital_alloc,
        sigma_c, expected_rc, weights, tickers,
        portfolio_info, sigma_p, mu, y_opt, mu_p, cov
    )
