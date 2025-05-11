import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def run(mu_real, cov_real, rf, mu_p, sigma_p, sigma_c, expected_rc, y_opt, y_capped):
    st.markdown("### Efficient Frontier with Simulated Portfolios")

    np.random.seed(42)
    n_sim = 3000
    n_assets = len(mu_real)

    # Simulate random weights
    weights_all = np.random.dirichlet(np.ones(n_assets), n_sim)
    returns_all = weights_all @ mu_real
    vols_all = np.sqrt(np.einsum('ij,ji->i', weights_all @ cov_real, weights_all.T))
    sharpes_all = (returns_all - rf) / vols_all

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#121212")
    sc = ax.scatter(vols_all * 100, returns_all * 100, c=sharpes_all, cmap='viridis', s=10, alpha=0.85)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Sharpe Ratio", color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.get_yticklabels(), color='white')

    # Risk-free point
    ax.scatter(0, rf * 100, c='blue', s=100, label=f"Risk-Free Rate ({rf*100:.2f}%)")

    # Optimal risky portfolio
    ax.scatter(sigma_p * 100, mu_p * 100, c='red', s=150, marker='*', label="Optimal Risky Portfolio")

    # CAL
    slope = (mu_p - rf) / sigma_p
    x_cal = np.linspace(0, vols_all.max() * 1.1, 100)
    y_cal = rf + slope * x_cal
    ax.plot(x_cal * 100, y_cal * 100, 'r--', linewidth=2, label="Capital Allocation Line (CAL)")

    # Complete portfolio
    ax.scatter(sigma_c * 100, expected_rc * 100, c='lime', marker='D', s=120,
               label=f"Complete Portfolio (y = {y_capped:.2f})")

    # Leveraged portfolio (optional)
    if abs(y_opt - y_capped) > 1e-3:
        sigma_uncap = y_opt * sigma_p
        ret_uncap = y_opt * mu_p + (1 - y_opt) * rf
        ax.scatter(sigma_uncap * 100, ret_uncap * 100, c='magenta', s=120, marker='X',
                   label=f"Leveraged Portfolio (y = {y_opt:.2f})")

    # Aesthetic
    ax.set_facecolor('#121212')
    fig.patch.set_facecolor('#121212')
    ax.set_title("Efficient Frontier with CAL and Optimal Portfolios", color='white')
    ax.set_xlabel("Volatility (%)", color='white')
    ax.set_ylabel("Expected Return (%)", color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1e1e1e', labelcolor='white', fontsize=9, loc='upper left')

    st.pyplot(fig)
