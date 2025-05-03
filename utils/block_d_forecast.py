def run(adj_returns_combinations, cov_matrix, selected_combo, rf, A):
    import numpy as np
    import pandas as pd
    import cvxpy as cp

    returns = np.array([adj_returns_combinations[selected_combo][t] for t in selected_combo]) / 100
    cov = cov_matrix.loc[selected_combo, selected_combo].values

    # Constraints
    n = len(selected_combo)
    w = cp.Variable(n)
    gamma = cp.Parameter(nonneg=True, value=10)
    beta_l2 = cp.Parameter(nonneg=True, value=0.05)

    portfolio_return = returns @ w
    portfolio_var = cp.quad_form(w, cov)

    # CVaR Soft Constraint Objective
    objective = cp.Maximize(portfolio_return - gamma * cp.norm(w, 1) - beta_l2 * cp.norm(w, 2))

    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= 0.5  # upper bound constraint per stock
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve()

    weights = w.value
    mu = portfolio_return.value
    sigma = np.sqrt(portfolio_var.value)

    # Capital Allocation Line - Utility Maximization
    y_opt = (mu - rf) / (A * sigma ** 2)
    y_capped = max(0, min(y_opt, 1))

    capital_risky = y_capped
    capital_rf = 1 - y_capped
    utility = mu - 0.5 * A * sigma ** 2

    allocation = {
        'Weights': dict(zip(selected_combo, weights)),
        'Expected Return': mu,
        'Volatility': sigma,
        'y*': y_capped,
        'Utility': utility,
        'Capital Allocation': {
            'Risky': capital_risky,
            'Risk-Free': capital_rf
        }
    }

    return allocation
