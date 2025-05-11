# utils/block_h3_visualization.py

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def run(hrp_result_dict, benchmark_return_mean, best_portfolio,
        mu_p, sigma_p, rf, sigma_c, expected_rc, y_capped, y_opt,
        tickers, weights, cov):

    st.markdown("### Efficient Frontier and Capital Allocation Line (CAL)")

    # === Simulate 1000 Random Portfolios ===
    n_sim = 1000
    np.random.seed(42)
    weights_random = np.random.dirichlet(np.ones(len(tickers)), n_sim)
    mu_vec = np.array([mu_p] * len(tickers))  # Optional, or adjust with actual vector
    returns = []
    risks = []
    sharpes = []

    for w in weights_random:
        mu_port = np.dot(w, mu_vec)
        sigma_port = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        sharpe = (mu_port - rf) / sigma_port if sigma_port > 0 else 0
        returns.append(mu_port * 100)
        risks.append(sigma_port * 100)
        sharpes.append(sharpe)

    returns = np.array(returns)
    risks = np.array(risks)
    sharpes = np.array(sharpes)

    # === Begin Plot ===
    fig, ax = plt.subplots(figsize=(10, 7), facecolor='#121212')

    sc = ax.scatter(risks, returns, c=sharpes, cmap='plasma', s=20,
                    edgecolors='none', alpha=0.9)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Sharpe Ratio', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.get_yticklabels(), color='white')

    # === Capital Allocation Line (CAL) ===
    slope = (mu_p - rf) / sigma_p
    x_cal = np.linspace(0, risks.max() * 1.2, 100)
    y_cal = rf + slope * (x_cal / 100)
    ax.plot(x_cal, y_cal * 100, 'r--', linewidth=2, label='Capital Allocation Line (CAL)')

    # === Risk-Free Point ===
    ax.scatter(0, rf * 100, color='blue', s=100, marker='o', label=f"Risk-Free Rate ({rf*100:.2f}%)")

    # === Optimal Risky Portfolio ===
    ax.scatter(sigma_p * 100, mu_p * 100, color='red', marker='*', s=180,
               label=f"Optimal Risky Portfolio")

    # === Complete Portfolio ===
    ax.scatter(sigma_c * 100, expected_rc * 100, color='lime', marker='D', s=160,
               label=f"Complete Portfolio (y = {y_capped:.2f})")

    # === Optional: Leveraged Portfolio if y_opt > y_capped ===
    if abs(y_opt - y_capped) > 1e-3:
        sigma_leveraged = y_opt * sigma_p
        mu_leveraged = y_opt * mu_p + (1 - y_opt) * rf
        ax.scatter(sigma_leveraged * 100, mu_leveraged * 100, color='magenta',
                   marker='X', s=150, label=f"Leveraged Portfolio (y = {y_opt:.2f})")

    # === Final Formatting ===
    ax.set_facecolor('#121212')
    fig.patch.set_facecolor('#121212')
    ax.set_title("Efficient Frontier with CAL and Optimal Portfolios", color='white')
    ax.set_xlabel("Volatility (%)", color='white')
    ax.set_ylabel("Expected Return (%)", color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1e1e1e', labelcolor='white', fontsize=9, loc='upper left')
    ax.grid(False)

    st.pyplot(fig)
