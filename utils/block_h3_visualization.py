import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

def run(best_portfolio, mu_p, sigma_p, rf, sigma_c, expected_rc, y_capped, y_opt,
        adj_returns_combinations=None, cov_matrix_dict=None, simulate_for_visual=True):

    st.markdown("### Efficient Frontier with Optimal Complete Portfolio")

    try:
        best_key = list(best_portfolio.keys())[0]
        tickers = list(best_key)
        result = best_portfolio[best_key]
        weights = np.array([result['Weights'][t] for t in tickers])
    except Exception as e:
        st.error(f"❌ Failed to extract best portfolio: {e}")
        return

    # Get real mu and cov
    if adj_returns_combinations and cov_matrix_dict:
        mu_dict = adj_returns_combinations[best_key]
        cov_df = cov_matrix_dict[best_key]
        mu_realistic = np.array([mu_dict[t] for t in tickers]) / 100
        cov = cov_df.loc[tickers, tickers].values
    else:
        mu_realistic = np.array([result['Expected Return (%)'] / 100] * len(tickers))
        cov = np.outer(weights, weights) * ((result['Volatility (%)'] / 100) ** 2)

    # Simulate random portfolios
    if simulate_for_visual:
        n_simulations = 10000
        np.random.seed(42)
        weights_sim = np.random.dirichlet(np.ones(len(tickers)), size=n_simulations)
        mu_sim = weights_sim @ mu_realistic
        sigma_sim = np.sqrt(np.einsum('ij,jk,ik->i', weights_sim, cov, weights_sim))
        sharpe_sim = (mu_sim - rf) / sigma_sim
        mask = (sigma_sim > 0.0001) & (mu_sim > 0.0001) & np.isfinite(sharpe_sim)
        mu_sim, sigma_sim, sharpe_sim = mu_sim[mask], sigma_sim[mask], sharpe_sim[mask]
    else:
        mu_sim, sigma_sim, sharpe_sim = [], [], []

    # === PLOT ===
    fig, ax = plt.subplots(figsize=(10, 7), dpi=100)

    # Efficient Frontier
    if simulate_for_visual and len(mu_sim) > 0:
        sc = ax.scatter(sigma_sim * 100, mu_sim * 100, c=sharpe_sim, cmap='viridis', s=15, alpha=0.85)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Sharpe Ratio", fontsize=11)

    # Markers
    ax.scatter(0, rf * 100, c='blue', s=100, label=f"Risk-Free Rate ({rf * 100:.2f}%)")
    ax.scatter(sigma_p * 100, mu_p * 100, c='red', marker='*', s=180,
               label=f"Optimal Risky Portfolio ({'-'.join(tickers)})")
    ax.scatter(sigma_c * 100, expected_rc * 100, c='green', marker='D', s=140,
               label=f"Optimal Complete Portfolio (y={y_capped:.2f})")

    # CAL line
    slope = (mu_p - rf) / sigma_p
    x_cal = np.linspace(0, sigma_sim.max() * 100 * 1.2, 200)
    y_cal = rf * 100 + slope * x_cal
    ax.plot(x_cal, y_cal, 'r--', linewidth=2, label="Capital Allocation Line (CAL)")

    # Styling
    ax.set_title("Efficient Frontier with Optimal Complete Portfolio", fontsize=14)
    ax.set_xlabel("Volatility (Risk) (%)", fontsize=12)
    ax.set_ylabel("Expected Return (%)", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.3)

    st.pyplot(fig)
