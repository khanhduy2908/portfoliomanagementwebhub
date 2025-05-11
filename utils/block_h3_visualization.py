import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

def run(best_portfolio, mu_p, sigma_p, rf, sigma_c, expected_rc, y_capped, y_opt,
        adj_returns_combinations=None, cov_matrix_dict=None, simulate_for_visual=True):

    st.markdown("### Efficient Frontier with CAL and Optimal Portfolios")

    try:
        best_key = list(best_portfolio.keys())[0]
        tickers = list(best_key)
        result = best_portfolio[best_key]
        weights = np.array([result['Weights'][t] for t in tickers])
    except Exception as e:
        st.error(f"❌ Failed to extract best portfolio: {e}")
        return

    # === Step 1: Get mu and cov realistically ===
    if adj_returns_combinations and cov_matrix_dict:
        try:
            mu_dict = adj_returns_combinations[best_key]
            cov_df = cov_matrix_dict[best_key]
            mu_realistic = np.array([mu_dict[t] for t in tickers]) / 100
            cov = cov_df.loc[tickers, tickers].values
        except Exception as e:
            st.warning(f"⚠️ Failed to fetch mu/cov from upstream, fallback to approximation: {e}")
            mu_realistic = np.array([result['Expected Return (%)'] / 100] * len(tickers))
            cov = np.outer(weights, weights) * ((result['Volatility (%)'] / 100) ** 2)
    else:
        st.warning("⚠️ adj_returns_combinations or cov_matrix_dict not provided. Using approximate values.")
        mu_realistic = np.array([result['Expected Return (%)'] / 100] * len(tickers))
        cov = np.outer(weights, weights) * ((result['Volatility (%)'] / 100) ** 2)

    # === Step 2: Simulate Efficient Frontier ===
    if simulate_for_visual:
        try:
            n_simulations = 10000
            np.random.seed(42)
            weights_sim = np.random.dirichlet(np.ones(len(tickers)), size=n_simulations)
            mu_sim = weights_sim @ mu_realistic
            sigma_sim = np.sqrt(np.einsum('ij,jk,ik->i', weights_sim, cov, weights_sim))
            sharpe_sim = (mu_sim - rf) / sigma_sim
            mask = (sigma_sim > 0.0001) & (mu_sim > 0.0001) & np.isfinite(sharpe_sim)
            mu_sim, sigma_sim, sharpe_sim = mu_sim[mask], sigma_sim[mask], sharpe_sim[mask]
        except Exception as e:
            st.warning(f"⚠️ Simulation failed: {e}")
            mu_sim, sigma_sim, sharpe_sim = [], [], []
    else:
        mu_sim, sigma_sim, sharpe_sim = [], [], []

    # === Step 3: Plot ===
    fig, ax = plt.subplots(figsize=(10, 7), facecolor="#121212")

    if simulate_for_visual and len(mu_sim) > 0:
        sc = ax.scatter(sigma_sim * 100, mu_sim * 100, c=sharpe_sim, cmap='viridis',
                        s=10, alpha=0.85, edgecolors='none')
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Sharpe Ratio", color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(cbar.ax.get_yticklabels(), color='white')

    # === Step 4: Key Portfolio Points ===
    ax.scatter(0, rf * 100, c='blue', s=100, label=f"Risk-Free Rate ({rf * 100:.2f}%)")
    ax.scatter(sigma_p * 100, mu_p * 100, c='red', marker='*', s=180, label="Optimal Risky Portfolio")
    ax.scatter(sigma_c * 100, expected_rc * 100, c='lime', marker='D', s=150,
               label=f"Complete Portfolio (y = {y_capped:.2f})")

    if abs(y_opt - y_capped) > 0.01:
        sigma_uncapped = y_opt * sigma_p
        rc_uncapped = y_opt * mu_p + (1 - y_opt) * rf
        ax.scatter(sigma_uncapped * 100, rc_uncapped * 100, c='magenta', marker='X', s=140,
                   label=f"Leveraged Portfolio (y = {y_opt:.2f})")

    # === Step 5: CAL Line ===
    slope = (mu_p - rf) / sigma_p
    max_x = max(sigma_sim.max() if len(sigma_sim) > 0 else sigma_p * 1.5, sigma_p * 1.5)
    x_cal = np.linspace(0, max_x, 100)
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
