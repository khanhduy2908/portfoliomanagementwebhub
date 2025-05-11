# utils/block_h3_visualization.py

import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

def run(hrp_result_dict, benchmark_return_mean, best_portfolio,
        mu_p, sigma_p, rf, sigma_c, expected_rc, y_capped, y_opt,
        tickers, weights, cov, n_simulated=2000):

    st.markdown("### Efficient Frontier and Capital Allocation Line (CAL)")

    np.random.seed(42)
    tickers = list(tickers)
    weights = np.array(weights)
    cov = np.array(cov)
    mu_p_pct = mu_p * 100
    sigma_p_pct = sigma_p * 100
    rf_pct = rf * 100

    # === 1. Mô phỏng portfolio ngẫu nhiên để tạo gradient ===
    sim_returns, sim_risks, sim_sharpes = [], [], []

    for _ in range(n_simulated):
        w = np.random.dirichlet(np.ones(len(tickers)))
        ret = np.dot(w, mu_pct)
        risk = np.sqrt(np.dot(w, np.dot(cov * 10000, w)))  # nhân để đưa về đơn vị %
        sharpe = (ret - benchmark_return_mean * 100) / risk if risk > 0 else np.nan

        sim_returns.append(ret)
        sim_risks.append(risk)
        sim_sharpes.append(sharpe)

    sim_returns = np.array(sim_returns)
    sim_risks = np.array(sim_risks)
    sim_sharpes = np.array(sim_sharpes)

    # Loại bỏ NaN
    mask = ~np.isnan(sim_sharpes)
    sim_returns = sim_returns[mask]
    sim_risks = sim_risks[mask]
    sim_sharpes = sim_sharpes[mask]

    # === 2. Vẽ biểu đồ ===
    fig, ax = plt.subplots(figsize=(10, 7), facecolor='#121212')

    # Gradient scatter
    scatter = ax.scatter(sim_risks, sim_returns, c=sim_sharpes, cmap='viridis',
                         s=12, alpha=0.8, edgecolors='none')

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Sharpe Ratio", color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.get_yticklabels(), color='white')

    # === 3. Risk-Free Point ===
    ax.scatter(0, rf_pct, c='blue', s=100, marker='o', label=f"Risk-Free Rate ({rf_pct:.2f}%)")

    # === 4. Optimal Risky Portfolio ===
    ax.scatter(sigma_p_pct, mu_p_pct, c='red', marker='*', s=200,
               label=f"Optimal Risky Portfolio ({'-'.join(tickers)})")

    # === 5. CAL Line ===
    slope = (mu_p - rf) / sigma_p
    x_cal = np.linspace(0, max(sim_risks.max(), sigma_p_pct * 1.3), 100)
    y_cal = rf_pct + slope * x_cal
    ax.plot(x_cal, y_cal, 'r--', linewidth=2, label="Capital Allocation Line (CAL)")

    # === 6. Complete Portfolio (Capped y) ===
    ax.scatter(sigma_c * 100, expected_rc * 100, c='lime', marker='D', s=160,
               label=f"Optimal Complete Portfolio (y={y_capped:.2f})")

    # === 7. Optional: Uncapped Portfolio y_opt > y_capped ===
    if abs(y_opt - y_capped) > 1e-3:
        sigma_uncapped = y_opt * sigma_p
        expected_uncapped = y_opt * mu_p + (1 - y_opt) * rf
        ax.scatter(sigma_uncapped * 100, expected_uncapped * 100,
                   c='magenta', marker='X', s=150,
                   label=f"Leveraged Portfolio (y={y_opt:.2f})")

    # === 8. Aesthetic ===
    ax.set_facecolor('#121212')
    fig.patch.set_facecolor('#121212')
    ax.set_title("Efficient Frontier with CAL and Optimal Portfolios", color='white', fontsize=14)
    ax.set_xlabel("Volatility (Risk) (%)", color='white')
    ax.set_ylabel("Expected Return (%)", color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1e1e1e', labelcolor='white', fontsize=9, loc='upper left')
    ax.grid(False)

    st.pyplot(fig)
