# utils/block_h3_visualization.py

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def run(mu, cov, rf, mu_p, sigma_p, sigma_c, expected_rc, y_capped, y_opt):
    st.markdown("### Efficient Frontier with Capital Allocation Line (CAL)")

    n_simulations = 3000
    np.random.seed(42)
    n_assets = len(mu)

    # Nhiễu đa dạng hoá
    mu_sim_all, sigma_sim_all, sharpe_sim_all = [], [], []

    for _ in range(n_simulations):
        w = np.random.dirichlet(np.ones(n_assets))
        mu_perturbed = mu + np.random.normal(0, 0.002, size=n_assets)
        cov_perturbed = cov + np.random.normal(0, 0.0005, size=cov.shape)
        cov_perturbed = (cov_perturbed + cov_perturbed.T) / 2  # Symmetrize

        try:
            ret = np.dot(w, mu_perturbed)
            vol = np.sqrt(w @ cov_perturbed @ w)
            sharpe = (ret - rf) / vol if vol > 0 else 0
            mu_sim_all.append(ret * 100)
            sigma_sim_all.append(vol * 100)
            sharpe_sim_all.append(sharpe)
        except:
            continue

    mu_sim_all = np.array(mu_sim_all)
    sigma_sim_all = np.array(sigma_sim_all)
    sharpe_sim_all = np.array(sharpe_sim_all)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7), facecolor="#121212")
    scatter = ax.scatter(sigma_sim_all, mu_sim_all, c=sharpe_sim_all, cmap='plasma', s=10, alpha=0.85)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Sharpe Ratio", color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.get_yticklabels(), color='white')

    # Risk-Free
    ax.scatter(0, rf * 100, c='blue', s=100, marker='o', label=f"Risk-Free Rate ({rf*100:.2f}%)")

    # Optimal Risky
    ax.scatter(sigma_p * 100, mu_p * 100, c='red', marker='*', s=250, label="Optimal Risky Portfolio")

    # CAL
    slope = (mu_p - rf) / sigma_p
    x_cal = np.linspace(0, max(sigma_sim_all.max(), sigma_p * 1.5), 100)
    y_cal = rf + slope * x_cal
    ax.plot(x_cal, y_cal * 100, 'r--', linewidth=2, label="Capital Allocation Line (CAL)")

    # Complete Portfolio
    ax.scatter(sigma_c * 100, expected_rc * 100, c='lime', marker='D', s=180,
               label=f"Complete Portfolio (y = {y_capped:.2f})")

    if abs(y_opt - y_capped) > 1e-3:
        sigma_leverage = y_opt * sigma_p
        mu_leverage = y_opt * mu_p + (1 - y_opt) * rf
        ax.scatter(sigma_leverage * 100, mu_leverage * 100,
                   c='magenta', marker='X', s=180,
                   label=f"Leveraged Portfolio (y = {y_opt:.2f})")

    ax.set_facecolor('#121212')
    fig.patch.set_facecolor('#121212')
    ax.set_title("Efficient Frontier with CAL and Optimal Portfolios", color='white', fontsize=14)
    ax.set_xlabel("Volatility (%)", color='white')
    ax.set_ylabel("Expected Return (%)", color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1e1e1e', labelcolor='white', fontsize=9, loc='upper left')
    ax.grid(False)

    st.pyplot(fig)
