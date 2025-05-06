# utils/block_h_complete_portfolio.py

import numpy as np
import pandas as pd
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

    portfolio_name = '-'.join(best_key) if isinstance(best_key, tuple) else str(best_key)
    mu = np.array([adj_returns_combinations[best_key][t] for t in tickers]) / 100
    cov = cov_matrix_dict[best_key].loc[tickers, tickers].values

    sigma_p = np.sqrt(weights.T @ cov @ weights)
    mu_p = np.dot(weights, mu)

    # --- Compute unconstrained y* ---
    y_opt = (mu_p - rf) / (A * sigma_p ** 2) if sigma_p > 0 else 0

    # --- Determine user risk profile and constraints ---
    score = config.risk_score  # passed from sidebar slider (10–40)
    if 10 <= score <= 17:
        risk_profile = 'low'
    elif 18 <= score <= 27:
        risk_profile = 'medium'
    elif 28 <= score <= 40:
        risk_profile = 'high'
    else:
        raise ValueError("Invalid risk score")

    profile_constraints = {
        'low':    {'y_min': 0.0, 'y_max': 0.4, 'min_rf_ratio': 0.6},
        'medium': {'y_min': 0.3, 'y_max': 0.7, 'min_rf_ratio': 0.3},
        'high':   {'y_min': 0.6, 'y_max': 1.0, 'min_rf_ratio': 0.0},
    }
    c = profile_constraints[risk_profile]

    # --- Clip y* based on constraints ---
    y_capped = float(np.clip(y_opt, c['y_min'], c['y_max']))
    capital_risky = y_capped * total_capital
    capital_rf = total_capital - capital_risky

    # --- Enforce risk-free minimum constraint ---
    rf_ratio = capital_rf / total_capital
    if rf_ratio < c['min_rf_ratio']:
        capital_rf = total_capital * c['min_rf_ratio']
        capital_risky = total_capital - capital_rf
        y_capped = capital_risky / total_capital

    expected_rc = y_capped * mu_p + (1 - y_capped) * rf
    sigma_c = y_capped * sigma_p
    utility = expected_rc - 0.5 * A * sigma_c ** 2

    capital_alloc = {t: capital_risky * w for t, w in zip(tickers, weights)}

    portfolio_info = {
        'portfolio_name': portfolio_name,
        'mu': mu_p,
        'sigma': sigma_p,
        'rf': rf,
        'y_opt': y_opt,
        'y_capped': y_capped,
        'A': A,
        'risk_score': score,
        'risk_profile': risk_profile,
        'expected_rc': expected_rc,
        'sigma_c': sigma_c,
        'utility': utility,
        'capital_risky': capital_risky,
        'capital_rf': capital_rf,
        'total_capital': total_capital,
    }

    return (best_portfolio, y_capped, capital_alloc,
            sigma_c, expected_rc, weights, tickers,
            portfolio_info, None, cov, mu, y_opt)
