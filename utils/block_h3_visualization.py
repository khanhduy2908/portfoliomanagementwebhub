import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd

def run(hrp_result_dict, benchmark_return_mean, results_ef, best_portfolio,
        mu_p, sigma_p, rf, sigma_c, expected_rc, y_capped, y_opt,
        tickers, weights, cov):

    st.markdown("### Efficient Frontier and Capital Allocation Line (CAL)")

    # === 1. Mô phỏng thêm danh mục để tạo gradient rõ ràng ===
    num_simulations = 3000
    np.random.seed(42)
    sim_returns = np.random.normal(loc=mu_p, scale=sigma_p, size=(num_simulations, len(tickers)))
    simulated_weights = np.random.dirichlet(np.ones(len(tickers)), num_simulations)
    
    returns_list, sigma_list, sharpe_list = [], [], []
    for w in simulated_weights:
        port_mu = w @ mu_p
        port_sigma = np.sqrt(w.T @ cov @ w)
        sharpe = (port_mu - rf) / port_sigma if port_sigma > 0 else 0
        returns_list.append(port_mu * 100)
        sigma_list.append(port_sigma * 100)
        sharpe_list.append(sharpe)

    # Thêm các điểm từ kết quả EF ban đầu
    returns_list.extend(results_ef[0])
    sigma_list.extend(results_ef[1])
    sharpe_list.extend(results_ef[2])

    returns_list = np.array(returns_list)
    sigma_list = np.array(sigma_list)
    sharpe_list = np.array(sharpe_list)

    # === 2. Vẽ biểu đồ ===
    fig, ax = plt.subplots(figsize=(10, 7), facecolor='#121212')

    scatter = ax.scatter(
        sigma_list, returns_list,
        c=sharpe_list, cmap='plasma',
        s=50, alpha=0.9, edgecolors='black', linewidths=0.3
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Sharpe Ratio", color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.get_yticklabels(), color='white')

    # === 3. Risk-Free & Optimal Points ===
    ax.scatter(0, rf * 100, c='blue', s=100, marker='o', label=f"Risk-Free ({rf*100:.2f}%)")
    ax.scatter(sigma_p * 100, mu_p * 100, c='red', s=160, marker='*', label="Optimal Risky Portfolio")
    ax.scatter(sigma_c * 100, expected_rc * 100, c='lime', s=150, marker='D',
               label=f"Complete Portfolio (y = {y_capped:.2f})")
    if abs(y_opt - y_capped) > 1e-3:
        sigma_uncapped = y_opt * sigma_p
        expected_uncapped = y_opt * mu_p + (1 - y_opt) * rf
        ax.scatter(sigma_uncapped * 100, expected_uncapped * 100,
                   c='purple', marker='D', s=150,
                   label=f"Complete Portfolio (y = {y_opt:.2f}, leveraged)")

    # === 4. CAL Line ===
    cal_x = np.linspace(0, max(sigma_list) * 1.2, 100)
    cal_y = rf + ((mu_p - rf) / sigma_p) * (cal_x / 100)
    ax.plot(cal_x, cal_y * 100, linestyle='--', color='red', linewidth=2, label="CAL Line")

    # === 5. Thẩm mỹ ===
    ax.set_title("Efficient Frontier with CAL and Optimal Portfolios", fontsize=14, color='white')
    ax.set_xlabel("Portfolio Volatility (%)", color='white')
    ax.set_ylabel("Expected Return (%)", color='white')
    ax.set_facecolor('#121212')
    fig.patch.set_facecolor('#121212')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1e1e1e', labelcolor='white', fontsize=9)
    ax.grid(False)

    st.pyplot(fig)
