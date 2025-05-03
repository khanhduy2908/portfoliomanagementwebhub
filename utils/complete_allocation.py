
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp

def construct_complete_portfolio(best_portfolio, adj_returns_combinations, cov_matrix_dict, A, rf, total_capital,
                                  alpha_cvar=0.95, lambda_cvar=10, beta_l2=0.05, n_simulations=30000,
                                  y_min=0.6, y_max=0.9):
    tickers = list(best_portfolio['Weights'].keys())
    weights_hrp = np.array(list(best_portfolio['Weights'].values()))

    mu = np.array([adj_returns_combinations[best_portfolio['Portfolio']][t] for t in tickers]) / 100
    cov = cov_matrix_dict[best_portfolio['Portfolio']].loc[tickers, tickers].values

    np.random.seed(42)
    simulated_returns = np.random.multivariate_normal(mean=mu, cov=cov, size=n_simulations)

    w = cp.Variable(len(tickers))
    VaR = cp.Variable()
    z = cp.Variable(n_simulations)

    port_returns = simulated_returns @ w
    loss = -port_returns
    cvar = VaR + cp.sum(z) / ((1 - alpha_cvar) * n_simulations)
    mean_ret = cp.sum(cp.multiply(mu, w))

    objective = cp.Maximize(mean_ret - lambda_cvar * cvar - beta_l2 * cp.sum_squares(w))
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        z >= 0,
        z >= loss - VaR
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver='SCS')

    if problem.status not in ['optimal', 'optimal_inaccurate'] or w.value is None:
        return None

    w_opt = w.value
    mu_p = float(mu @ w_opt)
    sigma_p = np.sqrt(w_opt.T @ cov @ w_opt)
    losses = -simulated_returns @ w_opt
    cvar_p = float(VaR.value + np.mean(np.maximum(losses - VaR.value, 0)) / (1 - alpha_cvar))

    y_opt = (mu_p - rf) / (A * sigma_p ** 2)
    y_capped = max(y_min, min(y_opt, y_max))

    expected_rc = y_capped * mu_p + (1 - y_capped) * rf
    sigma_c = y_capped * sigma_p
    U = expected_rc - 0.5 * A * sigma_c ** 2

    capital_risky = y_capped * total_capital
    capital_rf = total_capital - capital_risky
    capital_alloc = {tickers[i]: capital_risky * w_opt[i] for i in range(len(tickers))}

    return {
        'weights': dict(zip(tickers, w_opt)),
        'mu': mu_p,
        'sigma': sigma_p,
        'cvar': cvar_p,
        'y_opt': y_opt,
        'y_capped': y_capped,
        'expected_return_complete': expected_rc,
        'sigma_complete': sigma_c,
        'utility': U,
        'capital_rf': capital_rf,
        'capital_risky': capital_risky,
        'capital_allocation': capital_alloc
    }
