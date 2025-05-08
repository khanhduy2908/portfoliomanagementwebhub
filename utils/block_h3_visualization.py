# utils/block_h3_visualization.py

import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

def run(hrp_result_dict, benchmark_return_mean, results_ef, best_portfolio,
        mu_p, sigma_p, rf, sigma_c, expected_rc, y_capped, y_opt,
        tickers, weights, cov):

    st.markdown("### Efficient Frontier and Capital Allocation Line (CAL)")

    # 1. Chuẩn bị dữ liệu
    mu_list = np.array(results_ef[0])
    sigma_list = np.array(results_ef[1])
    sharpe_list = np.array(results_ef[2])

    if len(mu_list) == 0 or len(sigma_list) == 0:
        st.warning("No data available to plot the efficient frontier.")
        return

    # 2. Vẽ biểu đồ
    fig, ax = plt.subplots(figsize=(10, 7), facecolor='#121212')

    scatter = ax.scatter(
        sigma_list, mu_list,
        c=sharpe_list, cmap='plasma',
        s=60, alpha=0.95, edgecolors='black', linewidths=0.3
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Sharpe Ratio", color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.get_yticklabels(), color='white')

    # 3. Risk-free point
    ax.scatter(0, rf * 100, c='blue', s=90, marker='o', label=f"Risk-Free ({rf*100:.2f}%)")

    # 4. Optimal risky portfolio
    ax.scatter(sigma_p * 100, mu_p * 100, c='red', s=180, marker='*', label="Optimal Risky Portfolio")

    # 5. CAL line
    max_x = max(sigma_list.max(), sigma_p * 1.5)
    cal_x = np.linspace(0, max_x, 100)
    cal_y = rf + ((mu_p - rf) / sigma_p) * cal_x
    ax.plot(cal_x * 100, cal_y * 100, linestyle='--', color='red', linewidth=2, label="CAL Line")

    # 6. Optimal complete portfolio
    ax.scatter(sigma_c * 100, expected_rc * 100, c='lime', s=150, marker='D',
               label=f"Complete Portfolio (y = {y_capped:.2f})")

    # 7. Thẩm mỹ
    ax.set_title("Efficient Frontier with CAL and Optimal Portfolios", fontsize=14, color='white')
    ax.set_xlabel("Portfolio Volatility (%)", color='white')
    ax.set_ylabel("Expected Return (%)", color='white')
    ax.set_facecolor('#121212')
    fig.patch.set_facecolor('#121212')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1e1e1e', labelcolor='white', fontsize=9)
    ax.grid(False)

    st.pyplot(fig)
