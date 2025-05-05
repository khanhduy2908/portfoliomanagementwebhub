# utils/block_h_complete_portfolio.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp
import streamlit as st

# --- CONFIGURATION ---
alpha_cvar = 0.95
lambda_cvar = 10
beta_l2 = 0.05
n_simulations = 30000
y_min, y_max = 0.6, 0.9

def run(hrp_cvar_results, adj_returns_combinations, cov_matrix_dict, rf, A, total_capital):
    st.markdown("### 💼 Complete Portfolio Construction (CAL + Utility Optimization)")

    # --- Lấy danh mục tốt nhất ---
    if not hrp_cvar_results:
        st.error("❌ Không có danh mục hợp lệ từ Block G.")
        return None, None, None, None, None, None, None

    best_portfolio = max(hrp_cvar_results, key=lambda x: x['Sharpe Ratio'])
    tickers = list(best_portfolio['Weights'].keys())
    weights_hrp = np.array(list(best_portfolio['Weights'].values()))

    mu = np.array([adj_returns_combinations[best_portfolio['Portfolio']][t] for t in tickers]) / 100
    cov = cov_matrix_dict[best_portfolio['Portfolio']].loc[tickers, tickers].values

    # --- Simulate return scenarios ---
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
    constraints = [cp.sum(w) == 1, w >= 0, z >= 0, z >= loss - VaR]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver='SCS')

    if prob.status not in ['optimal', 'optimal_inaccurate'] or w.value is None:
        st.error("❌ Tối ưu hóa không thành công.")
        return None, None, None, None, None, None, None

    w_opt = w.value
    mu_p = float(mu @ w_opt)
    sigma_p = np.sqrt(w_opt.T @ cov @ w_opt)
    losses = -simulated_returns @ w_opt
    cvar_p = float(VaR.value + np.mean(np.maximum(losses - VaR.value, 0)) / (1 - alpha_cvar))

    # --- Capital Allocation Line ---
    y_opt = (mu_p - rf) / (A * sigma_p ** 2)
    y_capped = max(y_min, min(y_opt, y_max))

    expected_rc = y_capped * mu_p + (1 - y_capped) * rf
    sigma_c = y_capped * sigma_p
    U = expected_rc - 0.5 * A * sigma_c ** 2

    capital_risky = y_capped * total_capital
    capital_rf = total_capital - capital_risky
    capital_alloc = {tickers[i]: capital_risky * w_opt[i] for i in range(len(tickers))}

    # --- Display Summary ---
    st.markdown(f"#### ✅ Best Portfolio: `{best_portfolio['Portfolio']}`")
    st.write(f"• **Expected Return (E_rc)**: {expected_rc*100:.2f}%")
    st.write(f"• **Portfolio Volatility (σ_c)**: {sigma_c*100:.2f}%")
    st.write(f"• **Optimal y***: {y_opt:.2f} → Capped y: {y_capped:.2f}")
    st.write(f"• **Utility (U)**: {U:.4f}")
    st.write(f"• **Capital Allocation:** Risk-Free = {capital_rf:,.0f} VND | Risky = {capital_risky:,.0f} VND")

    for t, val in capital_alloc.items():
        st.write(f"  • {t}: {val:,.0f} VND")

    # --- Pie Chart ---
    sizes = [capital_rf] + [capital_alloc[t] for t in tickers]
    labels = ['Risk-Free'] + tickers
    if np.any(np.array(sizes) < 0):
        st.warning("⚠️ Không thể vẽ biểu đồ do giá trị âm.")
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, shadow=True, textprops={'fontsize': 12})
        ax.set_title("Final Capital Allocation")
        st.pyplot(fig)

    return best_portfolio, y_capped, capital_alloc, sigma_c, expected_rc, w_opt, tickers
