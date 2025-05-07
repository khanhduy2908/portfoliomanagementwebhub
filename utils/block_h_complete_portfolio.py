import numpy as np
from scipy.optimize import minimize_scalar

def run(hrp_result_dict, adj_returns_combinations, cov_matrix_dict,
        rf, A, total_capital, risk_score, y_min=0.6, y_max=0.9):

    if not hrp_result_dict:
        raise ValueError("❌ No valid HRP-CVaR portfolios from Block G.")

    # 1. Select portfolio with highest Sharpe
    best_key = max(hrp_result_dict, key=lambda k: hrp_result_dict[k]['Sharpe Ratio'])
    best_portfolio = hrp_result_dict[best_key]

    tickers = list(best_portfolio['Weights'].keys())
    weights = np.array([best_portfolio['Weights'][t] for t in tickers])
    weights /= weights.sum()

    mu_vec = np.array([adj_returns_combinations[best_key][t] for t in tickers]) / 100
    cov = cov_matrix_dict[best_key].loc[tickers, tickers].values

    mu_p = weights @ mu_vec
    sigma_p = np.sqrt(weights.T @ cov @ weights)

    if mu_p <= 0 or sigma_p <= 0:
        raise ValueError("❌ Selected portfolio has non-positive return or volatility.")

    # 2. Risk-free allocation constraint based on risk_score
    if 10 <= risk_score <= 17:
        max_rf_ratio = 0.40
    elif 18 <= risk_score <= 27:
        max_rf_ratio = 0.20
    elif 28 <= risk_score <= 40:
        max_rf_ratio = 0.10
    else:
        raise ValueError("Invalid risk score: must be between 10 and 40.")

    # 3. Capital allocation optimization: utility = E(Rc) - 0.5 * A * sigma_c^2
    def neg_utility(y):
        expected_rc = y * mu_p + (1 - y) * rf
        sigma_c = y * sigma_p
        return -(expected_rc - 0.5 * A * sigma_c**2)

    upper_bound = min(y_max, 1 - max_rf_ratio)
    if upper_bound <= y_min:
        raise ValueError("❌ Risk constraints too tight: upper_bound ≤ y_min.")

    opt_result = minimize_scalar(neg_utility, bounds=(y_min, upper_bound), method='bounded')
    y_opt = opt_result.x
    y_capped = np.clip(y_opt, y_min, upper_bound)

    # 4. Final complete portfolio metrics
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
        'A': A,
        'y_opt': y_opt,
        'y_capped': y_capped,
        'expected_rc': expected_rc,
        'sigma_c': sigma_c,
        'utility': utility,
        'capital_risky': capital_risky,
        'capital_rf': capital_rf
    }

    return (
        best_portfolio,       # portfolio dictionary with weights and metrics
        y_capped,             # final proportion y*
        capital_alloc,        # {ticker: allocated capital}
        sigma_c,              # final portfolio volatility
        expected_rc,          # expected return of complete portfolio
        weights,              # risky asset weights
        tickers,              # list of tickers in portfolio
        portfolio_info,       # summary dictionary
        sigma_p,              # portfolio volatility (risky only)
        mu_vec,               # expected return vector (risky)
        y_opt,                # optimal y before clipping
        mu_p,                 # expected return of risky portfolio
        cov                   # covariance matrix
    )
