# utils/block_h_complete_portfolio.py

import numpy as np
import pandas as pd
import warnings
from scipy.optimize import minimize_scalar
import config

def run(hrp_result_dict, adj_returns_combinations, cov_matrix_dict,
        rf, A, total_capital):

    if not hrp_result_dict:
        raise ValueError("No valid HRP-CVaR results from Block G.")

    # --- Select best portfolio by Sharpe Ratio ---
    best_key = max(hrp_result_dict, key=lambda k: hrp_result_dict[k]['Sharpe Ratio'])
    best_portfolio = hrp_result_dict[best_key]
    tickers = list(best_portfolio['Weights'].keys())
    weights = np.array(list(best_portfolio['Weights'].values()))

    if isinstance(best_key, tuple):
        portfolio_name = '-'.join(best_key)
    else:
        portfolio_name = str(best_key)

    mu = np.array([adj_returns_combinations[best_key][t] for t in tickers]) / 100
    cov = cov_matrix_dict[best_key].loc[tickers, tickers].values

    sigma_p = np.sqrt(weights.T @ cov @ weights)
    mu_p = np.dot(weights, mu)

    # --- Risk profile constraints (overwritten from app.py) ---
    y_min = getattr(config, 'y_min', 0.7)
    y_max = getattr(config, 'y_max', 0.95)

    # --- Utility Optimization within bounds ---
    def utility_neg(y):
        expected_rc = y * mu_p + (1 - y) * rf
        sigma_c = y * sigma_p
        utility = expected_rc - 0.5 * A * sigma_c**2
        return -utility

    opt_result = minimize_scalar(utility_neg, bounds=(y_min, y_max), method='bounded')
    y_opt = opt_result.x
    y_capped = np.clip(y_opt, y_min, y_max)

    expected_rc = y_capped * mu_p + (1 - y_capped) * rf
    sigma_c = y_capped * sigma_p
    utility = expected_rc - 0.5 * A * sigma_c ** 2

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

    # --- Warning if risk-free allocation is too high ---
    risk_free_ratio = capital_rf / total_capital
    if risk_free_ratio > 0.4:
        warnings.warn(f"Risk-Free allocation exceeded 40%: {risk_free_ratio:.2%}")

    return (best_portfolio, y_capped, capital_alloc,
            sigma_c, expected_rc, weights, tickers,
            portfolio_info, sigma_p, mu, y_opt, mu_p, cov)
