import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

def run(best_portfolio, mu_p, sigma_p, rf, sigma_c, expected_rc, y_capped, y_opt, tickers, weights, cov):
    st.markdown("### Efficient Frontier with CAL and Optimal Portfolios")

    # === Step 1: Simulate random portfolios from selected tickers ===
    n_simulations = 5000
    np.random.seed(42)
    weights_sim = np.random.dirichlet(np.ones(len(tickers)), size=n_simulations)

    # Estimate mean vector using the actual weights * mu_p as proxy
    mu_vec = np.array([mu_p / sum(weights)] * len(tickers))  # proxy assuming equal contribution to mu_p

    mu_sim = weights_sim @ mu_vec
    sigma_sim = np.sqrt(np.einsum('ij,jk,ik->i', weights_sim, cov, weights_sim))
    sharpe_sim = (mu_sim - rf) / sigma_sim

    # === Step 2: Filter portfolios to make curve cleaner ===
    filter_mask = (sigma_sim > 0.001) & (mu_sim > 0.001)
    mu_sim = mu_sim[filter_mask]
    sigma_sim = sigma_sim[filter_mask]
    sharpe_sim = sharpe_sim[filter_mask]

    # === Step 3: Plot Efficient Frontier ===
    fig, ax = plt.subplots(figsize=(10, 7), facecolor="#121212")

    sc = ax.scatter(
        sigma_sim * 100, mu_sim * 100,
        c=sharpe_sim,
        cmap='viridis',
        s=12,
        alpha=0.9,
        edgecolors='none'
    )

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Sharpe Ratio", color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.get_yticklabels(), color='white')

    # === Step 4: Important markers ===
    ax.scatter(0, rf * 100, c='blue', s=100, label=f"Risk-Free Rate ({rf * 100:.2f}%)")
    ax.scatter(sigma_p * 100, mu_p * 100, c='red', marker='*', s=180, label="Optimal Risky Portfolio")
    ax.scatter(sigma_c * 100, expected_rc * 100, c='lime', marker='D', s=150, label=f"Complete Portfolio (y = {y_capped:.2f})")

    if abs(y_opt - y_capped) > 0.01:
        sigma_uncapped = y_opt * sigma_p
        rc_uncapped = y_opt * mu_p + (1 - y_opt) * rf
        ax.scatter(sigma_uncapped * 100, rc_uncapped * 100, c='magenta', marker='X', s=140,
                   label=f"Leveraged Portfolio (y = {y_opt:.2f})")

    # === Step 5: CAL Line ===
    slope = (mu_p - rf) / sigma_p
    x_cal = np.linspace(0, sigma_sim.max() * 1.3, 100)
    y_cal = rf + slope * x_cal
    ax.plot(x_cal * 100, y_cal * 100, 'r--', linewidth=2, label="Capital Allocation Line (CAL)")

    # === Step 6: Styling ===
    ax.set_facecolor("#121212")
    fig.patch.set_facecolor("#121212")
    ax.set_title("Efficient Frontier with CAL and Optimal Portfolios", color='white')
    ax.set_xlabel("Volatility (%)", color='white')
    ax.set_ylabel("Expected Return (%)", color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1e1e1e', labelcolor='white', fontsize=9, loc='upper left')
    ax.grid(False)

    st.pyplot(fig)
