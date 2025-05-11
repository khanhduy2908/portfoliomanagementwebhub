
# utils/block_h3_visualization.py

import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

def run(hrp_result_dict, results_ef, best_portfolio,
        mu_p, sigma_p, rf, sigma_c, expected_rc, y_capped, y_opt):

    st.markdown("### Efficient Frontier and Capital Allocation Line (CAL)")

    mu_list = np.array(results_ef[0])
    sigma_list = np.array(results_ef[1])
    sharpe_list = np.array(results_ef[2])

    if len(mu_list) == 0 or len(sigma_list) == 0:
        st.warning("No data to plot efficient frontier.")
        return

    fig, ax = plt.subplots(figsize=(10, 7), facecolor='black')

    scatter = ax.scatter(sigma_list, mu_list, c=sharpe_list, cmap='cividis',
                         s=60, alpha=0.9, edgecolors='black', linewidths=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Sharpe Ratio", color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.get_yticklabels(), color='white')

    ax.scatter(0, rf * 100, c='deepskyblue', s=100, marker='o', label=f"Risk-Free ({rf*100:.2f}%)")
    ax.scatter(sigma_p * 100, mu_p * 100, c='red', s=200, marker='*',
               label=f"Optimal Risky Portfolio ({best_portfolio['Portfolio']})")

    slope = (mu_p - rf) / sigma_p
    x_cal = np.linspace(0, max(sigma_list) * 1.4, 100)
    y_cal = rf + slope * x_cal
    ax.plot(x_cal * 100, y_cal * 100, 'r--', linewidth=2, label="Capital Allocation Line (CAL)")

    ax.scatter(sigma_c * 100, expected_rc * 100, c='lime', s=150, marker='D',
               label=f"Complete Portfolio (y = {y_capped:.2f})")

    if abs(y_opt - y_capped) > 1e-3:
        sigma_leverage = y_opt * sigma_p
        expected_leverage = y_opt * mu_p + (1 - y_opt) * rf
        ax.scatter(sigma_leverage * 100, expected_leverage * 100,
                   c='violet', marker='D', s=150, label=f"Complete Portfolio (y = {y_opt:.2f}, leveraged)")

    ax.set_title("Efficient Frontier with CAL and Optimal Portfolios", color='white', fontsize=14)
    ax.set_xlabel("Portfolio Volatility (%)", color='white', fontsize=12)
    ax.set_ylabel("Expected Return (%)", color='white', fontsize=12)
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    ax.tick_params(colors='white')
    ax.legend(facecolor='black', labelcolor='white', fontsize=9)
    ax.grid(False)

    st.pyplot(fig)
