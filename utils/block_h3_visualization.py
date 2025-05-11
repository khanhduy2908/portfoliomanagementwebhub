# utils/block_h3_visualization.py

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def simulate_portfolios(mu, cov, rf, n_simulations=3000, seed=42):
    np.random.seed(seed)
    n_assets = len(mu)
    sim_returns, sim_risks, sim_sharpes = [], [], []

    for _ in range(n_simulations):
        weights = np.random.dirichlet(np.ones(n_assets))
        perturbed_mu = mu + np.random.normal(0, 0.002, size=n_assets)

        port_return = np.dot(weights, perturbed_mu)
        port_volatility = np.sqrt(weights.T @ cov @ weights)
        sharpe = (port_return - rf) / port_volatility if port_volatility > 0 else 0

        sim_returns.append(port_return * 100)
        sim_risks.append(port_volatility * 100)
        sim_sharpes.append(sharpe)

    return np.array(sim_risks), np.array(sim_returns), np.array(sim_sharpes)

def run(mu, cov, rf, mu_p, sigma_p, sigma_c, expected_rc, y_capped, y_opt):
    st.markdown("### Efficient Frontier with Capital Allocation Line (CAL)")

    # 1. Simulate portfolios
    sigma_list, mu_list, sharpe_list = simulate_portfolios(mu, cov, rf)

    fig, ax = plt.subplots(figsize=(10, 7), facecolor="#121212")
    scatter = ax.scatter(
        sigma_list, mu_list, c=sharpe_list, cmap='plasma',
        s=8, alpha=0.8, edgecolors='none'
    )

    # 2. Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Sharpe Ratio", color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.get_yticklabels(), color='white')

    # 3. Risk-free rate
    ax.scatter(0, rf * 100, c='blue', s=100, marker='o', label=f"Risk-Free Rate ({rf*100:.2f}%)")

    # 4. Optimal risky portfolio
    ax.scatter(sigma_p * 100, mu_p * 100, c='red', marker='*', s=250, label="Optimal Risky Portfolio")

    # 5. Capital Allocation Line
    slope = (mu_p - rf) / sigma_p
    x_cal = np.linspace(0, max(sigma_list.max(), sigma_p * 1.6), 100)
    y_cal = rf + slope * x_cal
    ax.plot(x_cal, y_cal * 100, 'r--', linewidth=2, label="Capital Allocation Line (CAL)")

    # 6. Complete Portfolio (y_capped)
    ax.scatter(sigma_c * 100, expected_rc * 100, c='lime', marker='D', s=180,
               label=f"Complete Portfolio (y = {y_capped:.2f})")

    # 7. Leveraged Portfolio (y_opt if y_opt > y_capped)
    if abs(y_opt - y_capped) > 1e-3:
        sigma_leverage = y_opt * sigma_p
        mu_leverage = y_opt * mu_p + (1 - y_opt) * rf
        ax.scatter(sigma_leverage * 100, mu_leverage * 100,
                   c='magenta', marker='X', s=180,
                   label=f"Leveraged Portfolio (y = {y_opt:.2f})")

    # 8. Aesthetic
    ax.set_facecolor('#121212')
    fig.patch.set_facecolor('#121212')
    ax.set_title("Efficient Frontier with CAL and Optimal Portfolios", color='white', fontsize=14)
    ax.set_xlabel("Volatility (%)", color='white')
    ax.set_ylabel("Expected Return (%)", color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1e1e1e', labelcolor='white', fontsize=9, loc='upper left')
    ax.grid(False)

    st.pyplot(fig)
