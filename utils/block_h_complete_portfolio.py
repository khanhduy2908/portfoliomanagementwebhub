# utils/block_h_complete_portfolio.py

import numpy as np
import config
from scipy.optimize import minimize_scalar

def run(hrp_result_dict, adj_returns_combinations, cov_matrix_dict,
        rf, A, total_capital):

    if not hrp_result_dict:
        raise ValueError("No valid HRP-CVaR results from Block G.")

    best_key = max(hrp_result_dict, key=lambda k: hrp_result_dict[k]['Sharpe Ratio'])
    best_portfolio = hrp_result_dict[best_key]
    tickers = list(best_portfolio['Weights'].keys())
    weights = np.array([best_portfolio['Weights'][t] for t in tickers])
    weights = weights / weights.sum()  # ensure weights are normalized

    # Portfolio metadata
    portfolio_name = '-'.join(best_key) if isinstance(best_key, tuple) else str(best_key)

    # Expected returns and covariance
    mu = np.array([adj_returns_combinations[best_key][t] for t in tickers]) / 100
    cov = cov_matrix_dict[best_key].loc[tickers, tickers].values

    sigma_p = np.sqrt(weights.T @ cov @ weights)
    mu_p = weights @ mu

    # Risk-free allocation constraint based on user's risk score
    score = getattr(config, 'risk_score', 25)
    if 10 <= score <= 17:
        max_rf_ratio = 0.40
    elif 18 <= score <= 27:
        max_rf_ratio = 0.20
    elif 28 <= score <= 40:
        max_rf_ratio = 0.10
    else:
        max_rf_ratio = 0.00

    def utility_neg(y):
        expected_rc = y * mu_p + (1 - y) * rf
        sigma_c = y * sigma_p
        return -(expected_rc - 0.5 * A * sigma_c**2)

    bounds = (0, 1 - max_rf_ratio)
    result = minimize_scalar(utility_neg, bounds=bounds, method='bounded')
    y_opt = result.x
    y_capped = np.clip(y_opt, config.y_min, config.y_max)

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
