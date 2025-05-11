# utils/block_h3_visualization.py

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

def run(hrp_result_dict, benchmark_return_mean,
        best_portfolio, mu_p, sigma_p, rf,
        sigma_c, expected_rc, y_capped, y_opt,
        tickers, weights, cov):

    st.markdown("### Efficient Frontier and Capital Allocation Line (CAL)")

    # === 1. Simulate additional portfolios ===
    np.random.seed(2024)
    n_simulated = 1000
    mu_arr = np.array(list(best_portfolio['Weights'].values())) * mu_p
    tickers = list(best_portfolio['Weights'].keys())
    w_base = np.array([best_portfolio['Weights'][t] for t in tickers])
    mu_sim, sigma_sim, sharpe_sim = [], [], []

    for _ in range(n_simulated):
        noise = np.random.normal(0, 0.02, len(w_base))
        w_rand = w_base + noise
        w_rand = np.clip(w_rand, 0, 1)
        w_rand /= w_rand.sum()

        mu_i = np.dot(w_rand, mu_arr) * 100
        sigma_i = np.sqrt(w_rand.T @ cov @ w_rand) * 100
        sharpe_i = (mu_i - rf * 100) / sigma_i if sigma_i > 0 else 0

        mu_sim.append(mu_i)
        sigma_sim.append(sigma_i)
        sharpe_sim.append(sharpe_i)

    mu_sim = np.array(mu_sim)
    sigma_sim = np.array(sigma_sim)
    sharpe_sim = np.array(sharpe_sim)

    # === 2. Plot Efficient Frontier ===
    fig, ax = plt.subplots(figsize=(10, 7), facecolor="#121212")
    sc = ax.scatter(sigma_sim, mu_sim, c=sharpe_sim, cmap='viridis', s=20, alpha=0.85)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Sharpe Ratio", color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.get_yticklabels(), color='white')

    # === 3. Risk-Free Rate ===
    ax.scatter(0, rf * 100, c='blue', s=100, marker='o', label=f"Risk-Free Rate ({rf*100:.2f}%)")

    # === 4. Optimal Risky Portfolio ===
    ax.scatter(sigma_p * 100, mu_p * 100, c='red', marker='*', s=200,
               label=f"Optimal Risky Portfolio ({'-'.join(tickers)})")

    # === 5. CAL Line ===
    slope = (mu_p - rf) / sigma_p
    x_cal = np.linspace(0, max(sigma_sim.max(), sigma_p * 1.5), 100)
    y_cal = rf + slope * x_cal
    ax.plot(x_cal, y_cal * 100, 'r--', linewidth=2, label="Capital Allocation Line (CAL)")

    # === 6. Optimal Complete Portfolio (Capped) ===
    ax.scatter(sigma_c * 100, expected_rc * 100, c='lime', marker='D', s=150,
               label=f"Complete Portfolio (y = {y_capped:.2f})")

    # === 7. Leveraged Portfolio (Uncapped y_opt) ===
    if abs(y_opt - y_capped) > 1e-3:
        sigma_uncapped = y_opt * sigma_p
        expected_uncapped = y_opt * mu_p + (1 - y_opt) * rf
        ax.scatter(sigma_uncapped * 100, expected_uncapped * 100,
                   c='magenta', marker='X', s=140,
                   label=f"Leveraged Portfolio (y = {y_opt:.2f})")

    # === 8. Aesthetic Settings ===
    ax.set_facecolor('#121212')
    fig.patch.set_facecolor('#121212')
    ax.set_title("Efficient Frontier with CAL and Optimal Portfolios", color='white')
    ax.set_xlabel("Volatility (%)", color='white')
    ax.set_ylabel("Expected Return (%)", color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1e1e1e', labelcolor='white', fontsize=9, loc='upper left')
    ax.grid(False)

    st.pyplot(fig)
