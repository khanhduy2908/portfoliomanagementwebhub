import numpy as np
import pandas as pd
import warnings

def run(hrp_result_dict, adj_returns_combinations, cov_matrix_dict,
        rf, A, total_capital, y_min=0.6, y_max=0.9):

    if not hrp_result_dict:
        raise ValueError("No valid HRP-CVaR results from Block G.")

    # --- Select best portfolio by Sharpe Ratio ---
    best_key = max(hrp_result_dict, key=lambda k: hrp_result_dict[k]['Sharpe Ratio'])
    best_portfolio = hrp_result_dict[best_key]
    tickers = list(best_portfolio['Weights'].keys())
    weights = np.array(list(best_portfolio['Weights'].values()))

    # Normalize portfolio name for display
    if isinstance(best_key, (tuple, list)):
        portfolio_name = '-'.join(best_key)
    else:
        portfolio_name = str(best_key)

    # --- Return & Covariance ---
    mu = np.array([adj_returns_combinations[best_key][t] for t in tickers]) / 100
    cov = cov_matrix_dict[best_key].loc[tickers, tickers].values

    # --- Optimal risky portfolio characteristics ---
    sigma_p = np.sqrt(weights.T @ cov @ weights)
    mu_p = weights @ mu

    # --- Compute y* and y capped ---
    y_opt = (mu_p - rf) / (A * sigma_p**2) if sigma_p > 0 else 0
    y_capped = np.clip(y_opt, y_min, y_max)

    # --- Complete portfolio stats ---
    expected_rc = y_capped * mu_p + (1 - y_capped) * rf
    sigma_c = y_capped * sigma_p
    utility = expected_rc - 0.5 * A * sigma_c**2

    # --- Capital allocation ---
    capital_risky = y_capped * total_capital
    capital_rf = total_capital - capital_risky
    capital_alloc = {t: capital_risky * w for t, w in zip(tickers, weights)}

    # --- Compile portfolio information for reporting ---
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
        'total_capital': total_capital
    }

    return (
        best_portfolio,
        y_capped,
        capital_alloc,
        sigma_c,
        expected_rc,
        weights,
        tickers,
        portfolio_info,
        None,
        cov,
        mu,
        y_opt
    )
