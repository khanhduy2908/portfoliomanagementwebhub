# utils/block_h3_visualization.py

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def run(best_portfolio, mu_p, sigma_p, rf, sigma_c, expected_rc, y_capped, y_opt, tickers, weights, cov):
    st.markdown("### Efficient Frontier with CAL and Optimal Portfolios")

    # 1. Mô phỏng các danh mục ngẫu nhiên dựa trên tickers và covariance
    n_portfolios = 3000
    np.random.seed(42)

    simulated_returns = []
    simulated_vols = []
    simulated_sharpes = []

    mu_vec = np.array(list(best_portfolio["Expected Return Vector"].values())) / 100
    tickers_ordered = list(best_portfolio["Weights"].keys())
    mu_vec = mu_vec[:len(tickers_ordered)]  # Đảm bảo đúng thứ tự
    cov_matrix = cov

    for _ in range(n_portfolios):
        w = np.random.dirichlet(np.ones(len(tickers_ordered)))
        port_return = np.dot(w, mu_vec)
        port_vol = np.sqrt(w.T @ cov_matrix @ w)
        sharpe = (port_return - rf) / port_vol if port_vol > 0 else 0

        simulated_returns.append(port_return * 100)
        simulated_vols.append(port_vol * 100)
        simulated_sharpes.append(sharpe)

    # 2. Vẽ biểu đồ với gradient Sharpe Ratio
    fig, ax = plt.subplots(figsize=(10, 7), facecolor="#121212")

    scatter = ax.scatter(
        simulated_vols, simulated_returns,
        c=simulated_sharpes,
        cmap='viridis',
        s=12,
        alpha=0.7,
        edgecolors='none'
    )

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Sharpe Ratio", color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.get_yticklabels(), color='white')

    # 3. Risk-Free Point
    ax.scatter(0, rf * 100, c='blue', s=100, marker='o', label=f"Risk-Free Rate ({rf*100:.2f}%)")

    # 4. Optimal Risky Portfolio
    ax.scatter(sigma_p * 100, mu_p * 100, c='red', marker='*', s=200, label="Optimal Risky Portfolio")

    # 5. Capital Allocation Line
    slope = (mu_p - rf) / sigma_p
    x_cal = np.linspace(0, max(simulated_vols) * 1.2, 100)
    y_cal = rf + slope * (x_cal / 100)
    ax.plot(x_cal, y_cal * 100, 'r--', linewidth=2, label="Capital Allocation Line (CAL)")

    # 6. Complete Portfolio (capped y)
    ax.scatter(sigma_c * 100, expected_rc * 100, c='lime', marker='D', s=140,
               label=f"Complete Portfolio (y = {y_capped:.2f})")

    # 7. Leveraged Portfolio nếu có y_opt > y_capped
    if abs(y_opt - y_capped) > 1e-3:
        sigma_uncapped = y_opt * sigma_p
        expected_uncapped = y_opt * mu_p + (1 - y_opt) * rf
        ax.scatter(sigma_uncapped * 100, expected_uncapped * 100,
                   c='magenta', marker='X', s=140,
                   label=f"Leveraged Portfolio (y = {y_opt:.2f})")

    # 8. Thẩm mỹ
    ax.set_facecolor("#121212")
    fig.patch.set_facecolor("#121212")
    ax.set_title("Efficient Frontier with CAL and Optimal Portfolios", color="white", fontsize=14)
    ax.set_xlabel("Volatility (%)", color="white")
    ax.set_ylabel("Expected Return (%)", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor='#1e1e1e', labelcolor='white', fontsize=9, loc='upper left')
    ax.grid(False)

    st.pyplot(fig)
