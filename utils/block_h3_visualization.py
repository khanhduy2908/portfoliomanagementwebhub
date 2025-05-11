# utils/block_h3_visualization.py

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

def run(
    best_portfolio, rf, mu_p, sigma_p, y_opt, y_capped, sigma_c, expected_rc,
    mu_sim=None, sigma_sim=None, sharpe_sim=None
):
    st.markdown("### Efficient Frontier with CAL and Optimal Portfolios")

    fig, ax = plt.subplots(figsize=(10, 7), facecolor="#121212")

    # === 1. Plot simulated portfolios if provided ===
    if mu_sim is not None and sigma_sim is not None and sharpe_sim is not None:
        mu_sim = np.array(mu_sim)
        sigma_sim = np.array(sigma_sim)
        sharpe_sim = np.array(sharpe_sim)

        mask = (~np.isnan(mu_sim)) & (~np.isnan(sigma_sim)) & (~np.isnan(sharpe_sim))
        mu_sim = mu_sim[mask]
        sigma_sim = sigma_sim[mask]
        sharpe_sim = sharpe_sim[mask]

        scatter = ax.scatter(
            sigma_sim, mu_sim,
            c=sharpe_sim,
            cmap="viridis",
            edgecolors='none',
            s=10,
            alpha=0.9
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Sharpe Ratio", color="white")
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(cbar.ax.get_yticklabels(), color='white')

    # === 2. Plot key points ===
    ax.scatter(0, rf * 100, c='blue', s=100, marker='o', label=f"Risk-Free Rate ({rf*100:.2f}%)")
    ax.scatter(sigma_p * 100, mu_p * 100, c='red', marker='*', s=200, label="Optimal Risky Portfolio")
    ax.scatter(sigma_c * 100, expected_rc * 100, c='lime', marker='D', s=150,
               label=f"Complete Portfolio (y = {y_capped:.2f})")

    if abs(y_opt - y_capped) > 1e-3:
        sigma_leverage = y_opt * sigma_p
        mu_leverage = y_opt * mu_p + (1 - y_opt) * rf
        ax.scatter(sigma_leverage * 100, mu_leverage * 100,
                   c='magenta', marker='X', s=150,
                   label=f"Leveraged Portfolio (y = {y_opt:.2f})")

    # === 3. Capital Allocation Line ===
    slope = (mu_p - rf) / sigma_p
    x_vals = np.linspace(0, max(sigma_p * 1.5, sigma_sim.max() * 1.1 if sigma_sim is not None else 0.3), 100)
    y_vals = rf + slope * x_vals
    ax.plot(x_vals * 100, y_vals * 100, 'r--', linewidth=2, label="Capital Allocation Line (CAL)")

    # === 4. Styling ===
    ax.set_facecolor("#121212")
    fig.patch.set_facecolor("#121212")
    ax.set_title("Efficient Frontier with CAL and Optimal Portfolios", color='white', fontsize=14)
    ax.set_xlabel("Volatility (%)", color='white')
    ax.set_ylabel("Expected Return (%)", color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1e1e1e', labelcolor='white', fontsize=9, loc='upper left')
    ax.grid(False)

    st.pyplot(fig)
