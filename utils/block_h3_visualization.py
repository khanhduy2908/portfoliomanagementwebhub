import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

def run(hrp_result_dict, benchmark_return_mean, best_portfolio,
        mu_p, sigma_p, rf, sigma_c, expected_rc, y_capped, y_opt,
        tickers, weights, cov):

    st.markdown("### Efficient Frontier and Capital Allocation Line (CAL)")

    # === 1. Simulate Random Portfolios ===
    np.random.seed(42)
    n_sim = 300
    mu = np.array([mu_p for _ in tickers])  # uniform expected returns
    sigma_matrix = cov

    sim_returns = []
    sim_vols = []
    sim_sharpes = []

    for _ in range(n_sim):
        w = np.random.dirichlet(np.ones(len(tickers)))
        ret = np.dot(w, mu) * 100
        vol = np.sqrt(w.T @ sigma_matrix @ w) * 100
        sharpe = (ret - rf * 100) / vol if vol > 0 else 0
        sim_returns.append(ret)
        sim_vols.append(vol)
        sim_sharpes.append(sharpe)

    # === 2. Plot Efficient Frontier with Gradient ===
    fig, ax = plt.subplots(figsize=(10, 7), facecolor="#121212")

    scatter = ax.scatter(
        sim_vols, sim_returns, c=sim_sharpes, cmap='plasma',
        s=50, edgecolors='black', linewidths=0.3, alpha=0.9
    )

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Sharpe Ratio", color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.get_yticklabels(), color='white')

    # === 3. Risk-Free Rate Point ===
    ax.scatter(0, rf * 100, c='blue', s=100, marker='o', label=f"Risk-Free Rate ({rf*100:.2f}%)")

    # === 4. Optimal Risky Portfolio ===
    ax.scatter(sigma_p * 100, mu_p * 100, c='red', marker='*', s=200, label="Optimal Risky Portfolio")

    # === 5. Capital Allocation Line (CAL) ===
    slope = (mu_p - rf) / sigma_p
    x_cal = np.linspace(0, max(max(sim_vols), sigma_p * 1.5), 100)
    y_cal = rf + slope * (x_cal / 100)
    ax.plot(x_cal, y_cal * 100, 'r--', linewidth=2, label="Capital Allocation Line (CAL)")

    # === 6. Complete Portfolio (capped y) ===
    ax.scatter(sigma_c * 100, expected_rc * 100, c='lime', marker='D', s=150,
               label=f"Complete Portfolio (y = {y_capped:.2f})")

    # === 7. Leveraged Portfolio nếu có y_opt > y_capped ===
    if abs(y_opt - y_capped) > 1e-3:
        sigma_uncapped = y_opt * sigma_p
        expected_uncapped = y_opt * mu_p + (1 - y_opt) * rf
        ax.scatter(sigma_uncapped * 100, expected_uncapped * 100,
                   c='magenta', marker='X', s=150,
                   label=f"Leveraged Portfolio (y = {y_opt:.2f})")

    # === 8. Aesthetics ===
    ax.set_facecolor('#121212')
    fig.patch.set_facecolor('#121212')
    ax.set_title("Efficient Frontier with CAL and Optimal Portfolios", color='white', fontsize=14)
    ax.set_xlabel("Volatility (%)", color='white')
    ax.set_ylabel("Expected Return (%)", color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1e1e1e', labelcolor='white', fontsize=9, loc='upper left')
    ax.grid(False)

    st.pyplot(fig)
