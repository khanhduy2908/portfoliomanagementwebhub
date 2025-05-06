# utils/block_h_complete_portfolio.py

import numpy as np
import cvxpy as cp
import pandas as pd
import streamlit as st

def run(hrp_cvar_results, adj_returns_combinations, cov_matrix_dict, rf, A, total_capital,
        alpha_cvar=0.95, lambda_cvar=10, beta_l2=0.05, n_simulations=30000,
        y_min=0.6, y_max=0.9):

    if not hrp_cvar_results:
        raise ValueError("No valid HRP-CVaR results from Block G.")

    # --- Select best portfolio by Sharpe Ratio ---
    best_key = max(hrp_cvar_results, key=lambda k: hrp_cvar_results[k]['Sharpe Ratio'])
    best_portfolio = hrp_cvar_results[best_key]
    tickers = list(best_portfolio['Weights'].keys())
    weights_hrp = np.array(list(best_portfolio['Weights'].values()))
    portfolio_name = best_key

    mu = np.array([adj_returns_combinations[portfolio_name][t] for t in tickers]) / 100
    cov = cov_matrix_dict[portfolio_name].loc[tickers, tickers].values

    # --- Simulate returns ---
    np.random.seed(42)
    simulated_returns = np.random.multivariate_normal(mean=mu, cov=cov, size=n_simulations)

    # --- CVaR Optimization ---
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
        raise ValueError("Complete portfolio optimization failed.")

    # --- Results ---
    w_opt = w.value
    mu_p = float(mu @ w_opt)
    sigma_p = np.sqrt(w_opt.T @ cov @ w_opt)
    losses = -simulated_returns @ w_opt
    cvar_p = float(VaR.value + np.mean(np.maximum(losses - VaR.value, 0)) / (1 - alpha_cvar))

    y_opt = (mu_p - rf) / (A * sigma_p**2)
    y_capped = max(y_min, min(y_opt, y_max))

    expected_rc = y_capped * mu_p + (1 - y_capped) * rf
    sigma_c = y_capped * sigma_p
    utility = expected_rc - 0.5 * A * sigma_c ** 2

    capital_risky = y_capped * total_capital
    capital_rf = total_capital - capital_risky
    capital_alloc = {tickers[i]: capital_risky * w_opt[i] for i in range(len(tickers))}

    # --- Return structured output ---
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
        "capital_risky": capital_risky,
        "capital_alloc": capital_alloc
    }

    st.subheader("Optimal Complete Portfolio Summary")
    st.markdown(f"**Selected Portfolio**: `{portfolio_name}`")
    st.markdown(f"- Risk Aversion (A): `{A}`")
    st.markdown(f"- Expected Return (E(rc)): `{expected_rc:.4f}`")
    st.markdown(f"- Portfolio Risk (σ_c): `{sigma_c:.4f}`")
    st.markdown(f"- Utility (U): `{utility:.4f}`")
    st.markdown(f"- Risk-Free Capital: `{capital_rf:,.0f} VND`")
    st.markdown(f"- Risky Capital: `{capital_risky:,.0f} VND`")
    st.markdown("**Capital Allocation to Risky Assets:**")

    alloc_df = pd.DataFrame({
        "Ticker": list(capital_alloc.keys()),
        "Allocated Capital (VND)": list(capital_alloc.values())
    })
    st.dataframe(alloc_df.style.format({"Allocated Capital (VND)": "{:,.0f}"}), use_container_width=True)


    return (
        best_portfolio,
        y_capped,
        capital_alloc,
        sigma_c,
        expected_rc,
        w_opt,
        tickers,
        portfolio_info,
        simulated_returns,
        cov,
        mu,
        y_opt
    )
