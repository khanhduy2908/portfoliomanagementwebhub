
import numpy as np
import cvxpy as cp

def run(hrp_cvar_results, adj_returns_combinations, cov_matrix_dict, rf, A, total_capital, risk_score=None):
    if not hrp_cvar_results:
        raise ValueError("No valid HRP-CVaR results from Block G.")

    best_key = max(hrp_cvar_results, key=lambda k: hrp_cvar_results[k]['Sharpe Ratio'])
    best_portfolio = hrp_cvar_results[best_key]
    tickers = list(best_portfolio['Weights'].keys())
    weights_hrp = np.array(list(best_portfolio['Weights'].values()))
    portfolio_name = best_key

    mu = np.array([adj_returns_combinations[portfolio_name][t] for t in tickers]) / 100
    cov = cov_matrix_dict[portfolio_name]

    # === Find optimal y* ===
    sigma_p = np.sqrt(weights_hrp.T @ cov @ weights_hrp)
    mu_p = weights_hrp @ mu
    y_opt = (mu_p - rf) / (A * sigma_p ** 2)

    # === Apply capped exposure based on risk tolerance ===
    if risk_score is None:
        max_rf_ratio = 0.25
    elif risk_score <= 17:
        max_rf_ratio = 0.40
    elif risk_score <= 27:
        max_rf_ratio = 0.25
    else:
        max_rf_ratio = 0.10

    y_capped = min(y_opt, 1 - max_rf_ratio)

    capital_risky = total_capital * y_capped
    capital_rf = total_capital - capital_risky

    capital_alloc = {t: w * capital_risky for t, w in zip(tickers, weights_hrp)}

    expected_rc = mu_p * y_capped + rf * (1 - y_capped)
    sigma_c = sigma_p * y_capped
    utility = expected_rc - 0.5 * A * sigma_c ** 2

    portfolio_info = {
        "portfolio_name": portfolio_name,
        "mu_p": mu_p,
        "sigma_p": sigma_p,
        "rf": rf,
        "y_opt": y_opt,
        "y_capped": y_capped,
        "A": A,
        "expected_rc": expected_rc,
        "sigma_c": sigma_c,
        "utility": utility,
        "capital_rf": capital_rf,
        "capital_risky": capital_risky
    }

    return best_portfolio, y_capped, capital_alloc, sigma_c, expected_rc, weights_hrp, tickers, portfolio_info, sigma_p, mu, y_opt, mu_p, cov
