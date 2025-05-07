# utils/block_h3_visualization.py

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np


def run(hrp_result_dict, benchmark_return_mean, results_ef, best_portfolio,
        mu_p, sigma_p, rf, sigma_c, expected_rc, y_capped, y_opt, tickers, cov):

    st.markdown("### Efficient Frontier with Optimal Complete Portfolio")

    # --- Simulate Efficient Frontier ---
    n_samples = 5000
    mu = np.array(results_ef[0])
    sigma = np.array(results_ef[1])
    sharpe = np.array(results_ef[2])

    if len(mu) < 3 or len(sigma) < 3:
        st.warning("Not enough portfolios to generate efficient frontier.")
        return

    fig, ax = plt.subplots(figsize=(12, 8), facecolor='black')
    scatter = ax.scatter(sigma * 100, mu * 100, c=sharpe, cmap='viridis', alpha=0.5, edgecolors='none')
    cbar = plt.colorbar(scatter, ax=ax, label="Sharpe Ratio")

    # --- Plot Optimal Risky Portfolio ---
    ax.scatter(sigma_p * 100, mu_p * 100, c='red', marker='*', s=250, label='Optimal Risky Portfolio')
    ax.scatter(0, rf * 100, c='blue', marker='o', s=150, label=f'Risk-Free Rate ({rf*100:.2f}%)')

    # --- CAL Line ---
    cal_x = np.linspace(0, max(sigma) * 1.5, 100)
    cal_y = rf + ((mu_p - rf) / sigma_p) * cal_x
    ax.plot(cal_x * 100, cal_y * 100, 'r--', label='Capital Allocation Line (CAL)')

    # --- Plot Optimal Complete Portfolio ---
    ax.scatter(sigma_c * 100, expected_rc * 100, c='lime', marker='D', s=200,
               label=f'Optimal Complete Portfolio (y={y_capped:.2f})')

    ax.set_title("Efficient Frontier with Optimal Complete Portfolio", color='white')
    ax.set_xlabel("Volatility (Risk) (%)", color='white')
    ax.set_ylabel("Expected Return (%)", color='white')
    ax.set_facecolor('black')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    ax.legend(facecolor='black', labelcolor='white')
    ax.grid(False)

    st.pyplot(fig)
    st.markdown("#### Selected Tickers")
    st.write(", ".join(tickers))
