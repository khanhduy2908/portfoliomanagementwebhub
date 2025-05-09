# utils/block_h3_visualization.py

import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd

def run(hrp_result_dict, benchmark_return_mean, results_ef, best_portfolio,
        mu_p, sigma_p, rf, sigma_c, expected_rc, y_capped, y_opt,
        tickers, weights, cov):

    st.markdown("### Efficient Frontier and Capital Allocation Line (CAL)")

    # 1. Tạo danh mục ngẫu nhiên để mô phỏng Efficient Frontier
    n_sim = 3000
    tickers = list(tickers)
    weights_arr = []
    returns_sim = []
    vol_sim = []
    sharpe_sim = []

    mu_vector = np.array([mu_p for _ in tickers])
    cov_matrix = cov

    for _ in range(n_sim):
        w = np.random.random(len(tickers))
        w /= w.sum()
        port_mu = w @ mu_vector
        port_sigma = np.sqrt(w.T @ cov_matrix @ w)
        sharpe = (port_mu - rf) / port_sigma if port_sigma > 0 else 0

        returns_sim.append(port_mu * 100)
        vol_sim.append(port_sigma * 100)
        sharpe_sim.append(sharpe)
        weights_arr.append(w)

    # 2. Vẽ biểu đồ
    fig, ax = plt.subplots(figsize=(10, 7), facecolor='#121212')

    scatter = ax.scatter(
        vol_sim, returns_sim,
        c=sharpe_sim, cmap='plasma',
        s=50, alpha=0.9, edgecolors='black', linewidths=0.3
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Sharpe Ratio", color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.get_yticklabels(), color='white')

    # 3. Risk-free
    ax.scatter(0, rf * 100, c='blue', s=100, marker='o', label=f"Risk-Free ({rf*100:.2f}%)")

    # 4. Optimal risky portfolio
    ax.scatter(sigma_p * 100, mu_p * 100, c='red', s=180, marker='*', label="Optimal Risky Portfolio")

    # 5. CAL
    max_x = max(max(vol_sim), sigma_p * 1.5)
    cal_x = np.linspace(0, max_x, 100)
    cal_y = rf + ((mu_p - rf) / sigma_p) * cal_x
    ax.plot(cal_x, cal_y * 100, linestyle='--', color='red', linewidth=2, label="CAL Line")

    # 6. Optimal complete portfolio
    ax.scatter(sigma_c * 100, expected_rc * 100, c='lime', s=150, marker='D',
               label=f"Complete Portfolio (y = {y_capped:.2f})")

    # 7. Nếu y_opt khác y_capped → hiển thị đòn bẩy
    if abs(y_opt - y_capped) > 1e-3:
        sigma_uncapped = y_opt * sigma_p
        expected_uncapped = y_opt * mu_p + (1 - y_opt) * rf
        ax.scatter(sigma_uncapped * 100, expected_uncapped * 100, c='purple', s=130, marker='D',
                   label=f"Complete Portfolio (y = {y_opt:.2f}, leveraged)")

    # 8. Aesthetic
    ax.set_title("Efficient Frontier with CAL and Optimal Portfolios", fontsize=14, color='white')
    ax.set_xlabel("Portfolio Volatility (%)", color='white')
    ax.set_ylabel("Expected Return (%)", color='white')
    ax.set_facecolor('#121212')
    fig.patch.set_facecolor('#121212')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1e1e1e', labelcolor='white', fontsize=9, loc='upper left')
    ax.grid(False)

    st.pyplot(fig)
