# utils/block_h3_visualization.py

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def run(best_portfolio, mu_p, sigma_p, rf, sigma_c, expected_rc, y_capped, y_opt, tickers, weights, cov):
    st.markdown("### Efficient Frontier with CAL and Optimal Portfolios")

    # === 1. Chuẩn bị thông số danh mục ===
    weights = np.array(weights)
    mu_vec = mu_p * np.ones_like(weights)
    cov_matrix = np.array(cov)

    # === 2. Mô phỏng danh mục ngẫu nhiên từ tổ hợp cổ phiếu thực tế ===
    n_portfolios = 3000
    np.random.seed(42)
    random_weights = np.random.dirichlet(np.ones(len(weights)), size=n_portfolios)

    port_returns = random_weights @ mu_vec
    port_vols = np.sqrt(np.sum(random_weights @ cov_matrix * random_weights, axis=1))
    sharpe_ratios = (port_returns - rf) / port_vols

    # === 3. Vẽ biểu đồ ===
    fig, ax = plt.subplots(figsize=(10, 7), facecolor='#121212')

    scatter = ax.scatter(
        port_vols * 100, port_returns * 100,
        c=sharpe_ratios,
        cmap='viridis',
        alpha=0.9,
        s=12,
        edgecolors='none'
    )

    # Colorbar
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Sharpe Ratio", color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.get_yticklabels(), color='white')

    # === 4. Vẽ các điểm đặc biệt ===
    ax.scatter(0, rf * 100, c='blue', s=80, marker='o', label=f"Risk-Free Rate ({rf*100:.2f}%)")
    ax.scatter(sigma_p * 100, mu_p * 100, c='red', s=130, marker='*', label="Optimal Risky Portfolio")
    ax.scatter(sigma_c * 100, expected_rc * 100, c='lime', s=100, marker='D', label=f"Complete Portfolio (y = {y_capped:.2f})")

    # Nếu có đòn bẩy thì vẽ thêm
    if abs(y_opt - y_capped) > 1e-3:
        sigma_lvg = y_opt * sigma_p
        mu_lvg = y_opt * mu_p + (1 - y_opt) * rf
        ax.scatter(sigma_lvg * 100, mu_lvg * 100, c='magenta', s=100, marker='X', label=f"Leveraged Portfolio (y = {y_opt:.2f})")

    # === 5. Vẽ CAL ===
    slope = (mu_p - rf) / sigma_p
    x_cal = np.linspace(0, (port_vols.max() + sigma_p * 0.5), 100)
    y_cal = rf + slope * x_cal
    ax.plot(x_cal * 100, y_cal * 100, 'r--', linewidth=2, label="Capital Allocation Line (CAL)")

    # === 6. Thẩm mỹ ===
    ax.set_facecolor('#121212')
    fig.patch.set_facecolor('#121212')
    ax.set_title("Efficient Frontier with CAL and Optimal Portfolios", color='white', fontsize=14)
    ax.set_xlabel("Volatility (%)", color='white')
    ax.set_ylabel("Expected Return (%)", color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1e1e1e', labelcolor='white', fontsize=9)
    ax.grid(False)

    st.pyplot(fig)
