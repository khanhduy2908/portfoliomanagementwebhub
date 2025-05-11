# utils/block_h3_visualization.py

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

def run(best_portfolio, mu_p, sigma_p, rf, sigma_c, expected_rc, y_capped, y_opt,
        tickers=None, weights=None, cov=None, simulate_for_visual=True):

    st.markdown("### Efficient Frontier with CAL and Optimal Portfolios")

    # === Step 1: Extract best portfolio details from dict ===
    if isinstance(best_portfolio, dict):
        try:
            # Chọn danh mục đầu tiên (vì bạn dùng fallback đã là danh mục tốt nhất)
            key = list(best_portfolio.keys())[0]
            result = best_portfolio[key]

            tickers = list(key)
            weights = np.array([result['Weights'][t] for t in tickers])
            mu_realistic = np.array([result['Expected Return (%)'] / 100] * len(tickers))
            cov = np.outer(weights, weights) * ((result['Volatility (%)'] / 100) ** 2)
        except Exception as e:
            st.error(f"❌ Failed to extract best portfolio: {e}")
            return
    else:
        st.error("❌ best_portfolio is not in expected dictionary format.")
        return

    # === Step 2: Simulate efficient frontier ===
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

    ax.scatter(0, rf * 100, c='blue', s=100, label=f"Risk-Free Rate ({rf * 100:.2f}%)")
    ax.scatter(sigma_p * 100, mu_p * 100, c='red', marker='*', s=180, label="Optimal Risky Portfolio")
    ax.scatter(sigma_c * 100, expected_rc * 100, c='lime', marker='D', s=150,
               label=f"Complete Portfolio (y = {y_capped:.2f})")

    if abs(y_opt - y_capped) > 0.01:
        sigma_uncapped = y_opt * sigma_p
        rc_uncapped = y_opt * mu_p + (1 - y_opt) * rf
        ax.scatter(sigma_uncapped * 100, rc_uncapped * 100, c='magenta', marker='X', s=140,
                   label=f"Leveraged Portfolio (y = {y_opt:.2f})")

    slope = (mu_p - rf) / sigma_p
    x_cal = np.linspace(0, max(sigma_sim.max() if len(sigma_sim) > 0 else sigma_p * 1.5, sigma_p * 1.5), 100)
    y_cal = rf + slope * x_cal
    ax.plot(x_cal * 100, y_cal * 100, 'r--', linewidth=2, label="Capital Allocation Line (CAL)")

    ax.set_facecolor("#121212")
    fig.patch.set_facecolor("#121212")
    ax.set_title("Efficient Frontier with CAL and Optimal Portfolios", color='white')
    ax.set_xlabel("Volatility (%)", color='white')
    ax.set_ylabel("Expected Return (%)", color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1e1e1e', labelcolor='white', fontsize=9, loc='upper left')
    ax.grid(False)

    st.pyplot(fig)
