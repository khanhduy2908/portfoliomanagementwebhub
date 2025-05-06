import numpy as np
import cvxpy as cp
import warnings

def run(hrp_cvar_results, adj_returns_combinations, cov_matrix_dict, rf, A, total_capital,
        alpha_cvar=0.95, lambda_cvar=10, beta_l2=0.05, n_simulations=30000,
        y_min=0.6, y_max=0.9):
    
    if not hrp_cvar_results:
        raise ValueError("No valid HRP-CVaR results from Block G.")

    # --- Select best portfolio by Sharpe Ratio ---
    best_key = max(hrp_cvar_results, key=lambda k: hrp_cvar_results[k]['Sharpe Ratio'])
    best_portfolio = hrp_cvar_results[best_key]
    tickers = list(best_portfolio['Weights'].keys())
    weights = np.array(list(best_portfolio['Weights'].values()))
    portfolio_name = best_key  # tuple

    # --- Expected returns & covariance ---
    mu = np.array([adj_returns_combinations[portfolio_name][t] for t in tickers]) / 100
    cov = cov_matrix_dict[portfolio_name].loc[tickers, tickers].values

    # --- Simulate returns ---
    np.random.seed(42)
    simulated_returns = np.random.multivariate_normal(mean=mu, cov=cov, size=n_simulations)

    portfolio_returns = simulated_returns @ weights
    expected_rc = np.mean(portfolio_returns)
    sigma_c = np.std(portfolio_returns)

    # --- Compute optimal capital allocation y* ---
    y_opt = (expected_rc - rf) / (A * sigma_c ** 2)
    y_capped = max(min(y_opt, y_max), y_min)

    # --- Final capital allocation ---
    capital_risky = y_capped * total_capital
    capital_rf = total_capital - capital_risky

    capital_alloc = {
        t: round(w * capital_risky, 0) for t, w in zip(tickers, weights)
    }

    utility = y_capped * (expected_rc - rf) - 0.5 * A * (y_capped * sigma_c) ** 2

    portfolio_info = {
        "Portfolio Name": "-".join(tickers),
        "Expected Return": expected_rc,
        "Portfolio Volatility": sigma_c,
        "Risk-Free Rate": rf,
        "Risk Aversion (A)": A,
        "y_opt": y_opt,
        "y_capped": y_capped,
        "Total Capital": total_capital,
        "Capital Risky": capital_risky,
        "Capital Risk-Free": capital_rf,
        "Utility": utility
    }

    return (
        best_portfolio, y_capped, capital_alloc, sigma_c, expected_rc,
        weights, tickers, portfolio_info, simulated_returns, cov, mu, y_opt
    )
