# utils/block_h3_visualization.py

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def run(hrp_result_dict, benchmark_return_mean, results_ef, best_portfolio,
        mu_p, sigma_p, rf, sigma_c, expected_rc, y_capped, y_opt,
        tickers, weights, cov, n_sim=3000):

    st.markdown("### Efficient Frontier and Capital Allocation Line (CAL)")

    # === Simulate Efficient Frontier from best portfolio's mu & cov ===
    def simulate_portfolios(mu_vec, cov_matrix, rf, n_portfolios=3000):
        n_assets = len(mu_vec)
        results = np.zeros((3, n_portfolios))
        for i in range(n_portfolios):
            w = np.random.rand(n_assets)
            w /= np.sum(w)
            ret = np.dot(w, mu_vec)
            vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
            sharpe = (ret - rf) / vol if vol > 0 else 0
            results[0, i] = ret * 100
            results[1, i] = vol * 100
            results[2, i] = sharpe
        return results

    mu_vec = np.array([mu_p] * len(tickers)) if isinstance(mu_p, float) else mu_p
    ef_results = simulate_portfolios(mu_vec, cov, rf, n_sim)

    # === Plot ===
    fig, ax = plt.subplots(figsize=(10, 7), facecolor='#121212')

    scatter = ax.scatter(
        ef_results[1], ef_results[0], c=ef_results[2],
        cmap='plasma', s=40, alpha=0.9, edgecolors='black', linewidths=0.3, label='Simulated Portfolios'
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Sharpe Ratio", color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.get_yticklabels(), color='white')

    # Risk-free point
    ax.scatter(0, rf * 100, c='blue', s=100, marker='o', label=f"Risk-Free Rate ({rf*100:.2f}%)")

    # Optimal Risky Portfolio
    ax.scatter(sigma_p * 100, mu_p * 100, c='red', s=200, marker='*', label="Optimal Risky Portfolio")

    # CAL Line
    cal_x = np.linspace(0, ef_results[1].max() * 1.2, 100)
    cal_y = rf + ((mu_p - rf) / sigma_p) * (cal_x / 100)
    ax.plot(cal_x, cal_y * 100, color='red', linestyle='--', linewidth=2, label="CAL Line")

    # Complete Portfolio
    ax.scatter(sigma_c * 100, expected_rc * 100, c='lime', s=150, marker='D',
               label=f"Complete Portfolio (y={y_capped:.2f})")

    if abs(y_opt - y_capped) > 1e-3:
        sigma_leveraged = y_opt * sigma_p
        return_leveraged = y_opt * mu_p + (1 - y_opt) * rf
        ax.scatter(sigma_leveraged * 100, return_leveraged * 100, c='purple', s=150, marker='D',
                   label=f"Leveraged (y={y_opt:.2f})")

    # Final polish
    ax.set_title("Efficient Frontier with CAL and Optimal Portfolios", fontsize=14, color='white')
    ax.set_xlabel("Portfolio Volatility (%)", color='white')
    ax.set_ylabel("Expected Return (%)", color='white')
    ax.set_facecolor('#121212')
    fig.patch.set_facecolor('#121212')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1e1e1e', labelcolor='white', fontsize=9)
    ax.grid(False)

    st.pyplot(fig)
