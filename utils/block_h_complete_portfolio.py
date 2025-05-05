import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp
from numpy.linalg import LinAlgError
from scipy.stats import multivariate_normal

import config

def run(hrp_cvar_results, adj_returns_combinations, cov_matrix_dict, rf_monthly, A, total_capital):
    print("Block H: Constructing Complete Portfolio (CAL + Utility)...")

    if not hrp_cvar_results:
        raise ValueError("No optimized portfolios found in Block G.")

    # --- Select best portfolio (highest Sharpe) ---
    best_portfolio = max(hrp_cvar_results, key=lambda x: x['Sharpe Ratio'])
    tickers = list(best_portfolio['Weights'].keys())
    weights_hrp = np.array(list(best_portfolio['Weights'].values()))
    portfolio_name = best_portfolio['Portfolio']

    mu = np.array([adj_returns_combinations[portfolio_name][t] for t in tickers]) / 100
    cov = cov_matrix_dict[portfolio_name].loc[tickers, tickers].values

    # --- Simulate Return Scenarios ---
    np.random.seed(config.SEED)
    simulated_returns = np.random.multivariate_normal(mean=mu, cov=cov, size=config.N_SIMULATIONS)

    # --- Optimization Variables ---
    w = cp.Variable(len(tickers))
    VaR = cp.Variable()
    z = cp.Variable(config.N_SIMULATIONS)

    port_returns = simulated_returns @ w
    loss = -port_returns
    cvar = VaR + cp.sum(z) / ((1 - config.CVaR_ALPHA) * config.N_SIMULATIONS)
    mean_ret = cp.sum(cp.multiply(mu, w))

    objective = cp.Maximize(mean_ret - config.LAMBDA_CVaR * cvar - config.BETA_L2 * cp.sum_squares(w))
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        z >= 0,
        z >= loss - VaR
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=config.SOLVERS[0])

    if prob.status not in ['optimal', 'optimal_inaccurate'] or w.value is None:
        raise ValueError("Optimization for complete portfolio failed.")

    w_opt = w.value
    mu_p = float(mu @ w_opt)
    sigma_p = float(np.sqrt(w_opt.T @ cov @ w_opt))

    losses = -simulated_returns @ w_opt
    cvar_p = float(VaR.value + np.mean(np.maximum(losses - VaR.value, 0)) / (1 - config.CVaR_ALPHA))

    # --- Capital Allocation Line (CAL) ---
    y_opt = (mu_p - rf_monthly) / (A * sigma_p ** 2)
    y_capped = max(config.Y_MIN, min(y_opt, config.Y_MAX))

    expected_rc = y_capped * mu_p + (1 - y_capped) * rf_monthly
    sigma_c = y_capped * sigma_p
    utility = expected_rc - 0.5 * A * sigma_c ** 2

    capital_risky = y_capped * total_capital
    capital_rf = total_capital - capital_risky
    capital_alloc = {tickers[i]: capital_risky * w_opt[i] for i in range(len(tickers))}

    # --- Logging Summary ---
    print(f"\nSelected Portfolio: {portfolio_name}")
    print(f"[CHECK] μ: {mu_p:.4f}, σ: {sigma_p * 100:.4f}%, rf: {rf_monthly:.4f}")
    print(f"[CHECK] y_opt: {y_opt:.4f}, final y: {y_capped:.4f}")
    print(f"Risk Aversion (A): {A}")
    print(f"Expected Return (E_rc): {expected_rc:.4f}")
    print(f"Portfolio Risk (σ_c): {sigma_c:.4f}")
    print(f"Utility (U): {utility:.4f}")
    print(f"Capital: Risk-Free = {capital_rf:,.0f} VND | Risky = {capital_risky:,.0f} VND")

    for t, val in capital_alloc.items():
        print(f"   • {t}: {val:,.0f} VND")

    # --- Pie Chart ---
    labels = ['Risk-Free Asset'] + tickers
    sizes = [capital_rf] + [capital_alloc[t] for t in tickers]

    if not np.any(np.array(sizes) < 0):
        plt.figure(figsize=(8, 6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%',
                startangle=90, shadow=True, textprops={'fontsize': 12})
        plt.title("Optimal Complete Portfolio Allocation", fontsize=14)
        plt.tight_layout()
        plt.show()

    return best_portfolio, y_capped, capital_alloc, sigma_c, expected_rc, w_opt, tickers
