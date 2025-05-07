# utils/block_h_complete_portfolio.py

import numpy as np
from scipy.optimize import minimize_scalar

def get_max_rf_ratio(risk_score):
    if 10 <= risk_score <= 17:
        return 0.40
    elif 18 <= risk_score <= 27:
        return 0.20
    elif 28 <= risk_score <= 40:
        return 0.10
    else:
        raise ValueError("Invalid risk score: must be between 10 and 40.")

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

    # 2. Return vector & Covariance matrix
    mu = np.array([adj_returns_combinations[best_key][t] for t in tickers]) / 100
    try:
        cov = cov_matrix_dict[best_key].loc[tickers, tickers].values
    except KeyError:
        raise ValueError("Covariance matrix does not contain all selected tickers.")

    sigma_p = np.sqrt(weights.T @ cov @ weights)
    mu_p = weights @ mu

    # 3. Check for validity
    if mu_p <= 0 or sigma_p <= 0:
        raise ValueError("Selected portfolio has non-positive return or volatility.")

    # 4. Risk-free allocation constraint
    max_rf_ratio = get_max_rf_ratio(risk_score)
    upper_bound = min(y_max, 1 - max_rf_ratio)

    if upper_bound <= y_min:
        raise ValueError("Risk constraints too tight: upper_bound <= y_min")

    # 5. Utility optimization
    def neg_utility(y):
        expected_rc = y * mu_p + (1 - y) * rf
        sigma_c = y * sigma_p
        return -(expected_rc - 0.5 * A * sigma_c**2)

    result = minimize_scalar(neg_utility, bounds=(y_min, upper_bound), method='bounded')
    y_opt = result.x
    y_capped = np.clip(y_opt, y_min, upper_bound)

    # 6. Portfolio metrics
    expected_rc = y_capped * mu_p + (1 - y_capped) * rf
    sigma_c = y_capped * sigma_p
    utility = expected_rc - 0.5 * A * sigma_c**2

    capital_risky = y_capped * total_capital
    capital_rf = total_capital - capital_risky
    capital_alloc = {t: capital_risky * w for t, w in zip(tickers, weights)}

    portfolio_info = {
        'portfolio_name': '-'.join(best_key) if isinstance(best_key, tuple) else str(best_key),
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
        'risk_score': risk_score
    }

    return (
        best_portfolio,         # dict with 'Weights'
        y_capped,               # capital exposure to risky assets
        capital_alloc,          # dict of capital per ticker
        sigma_c,                # std of complete portfolio
        expected_rc,            # expected return of complete portfolio
        weights,                # np.array of weights
        tickers,                # list of tickers
        portfolio_info,         # dict of summary info
        sigma_p,                # std of risky portfolio
        mu,                     # np.array of expected returns
        y_opt,                  # original optimal y before capping
        mu_p,                   # expected return of risky portfolio
        cov                     # covariance matrix
    )
