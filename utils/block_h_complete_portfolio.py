# utils/block_h_complete_portfolio.py

import numpy as np
import cvxpy as cp

def run(hrp_cvar_results, adj_returns_combinations, cov_matrix_dict, rf, A, total_capital,
        alpha_cvar=0.95, lambda_cvar=10, beta_l2=0.05, n_simulations=30000,
        y_min=0.6, y_max=0.9):

    if not hrp_cvar_results:
        raise ValueError("❌ No valid HRP-CVaR results from Block G.")

    # --- Chọn danh mục tốt nhất theo Sharpe Ratio ---
    best_portfolio = max(hrp_cvar_results, key=lambda x: x['Sharpe Ratio'])
    tickers = list(best_portfolio['Weights'].keys())
    weights_hrp = np.array(list(best_portfolio['Weights'].values()))
    portfolio_name = best_portfolio['Portfolio']

    mu = np.array([adj_returns_combinations[portfolio_name][t] for t in tickers]) / 100
    cov = cov_matrix_dict[portfolio_name].loc[tickers, tickers].values

    # --- Simulate Scenarios ---
    np.random.seed(42)
    simulated_returns = np.random.multivariate_normal(mean=mu, cov=cov, size=n_simulations)

    # --- Optimization Variables ---
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
        raise ValueError("❌ Complete portfolio optimization failed.")

    # --- Extract Optimal Portfolio Info ---
    w_opt = w.value
    mu_p = float(mu @ w_opt)
    sigma_p = np.sqrt(w_opt.T @ cov @ w_opt)
    losses = -simulated_returns @ w_opt
    cvar_p = float(VaR.value + np.mean(np.maximum(losses - VaR.value, 0)) / (1 - alpha_cvar))

    # --- Capital Allocation: y* ---
    y_opt = (mu_p - rf) / (A * sigma_p**2)
    y_capped = max(y_min, min(y_opt, y_max))

    expected_rc = y_capped * mu_p + (1 - y_capped) * rf
    sigma_c = y_capped * sigma_p
    utility = expected_rc - 0.5 * A * sigma_c ** 2

    capital_risky = y_capped * total_capital
    capital_rf = total_capital - capital_risky
    capital_alloc = {tickers[i]: capital_risky * w_opt[i] for i in range(len(tickers))}

    # --- Logging ---
    print(f"\n📌 Selected Portfolio: {portfolio_name}")
    print(f"[CHECK] mu: {mu_p:.4f}, sigma: {sigma_p * 100:.4f}%, rf: {rf:.4f}")
    print(f"[CHECK] y_opt: {y_opt:.4f}, final y: {y_capped:.4f}")
    print(f"🎯 Risk Aversion (A): {A}")
    print(f"📊 Expected Return (E_rc): {expected_rc:.4f}")
    print(f"📉 Portfolio Risk (σ_c): {sigma_c:.4f}")
    print(f"💡 Utility (U): {utility:.4f}")
    print(f"💰 Capital: Risk-Free = {capital_rf:,.0f} VND | Risky = {capital_risky:,.0f} VND")
    for t, val in capital_alloc.items():
        print(f"   • {t}: {val:,.0f} VND")

    return (
        best_portfolio,
        y_capped,
        capital_alloc,
        sigma_c,
        expected_rc,
        w_opt,
        tickers
    )
