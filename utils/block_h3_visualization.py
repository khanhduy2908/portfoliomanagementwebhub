import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def run(hrp_result_dict, benchmark_return_mean, results_ef, best_portfolio,
        mu_p, sigma_p, rf, sigma_c, expected_rc, y_capped, y_opt, tickers, cov):

    st.markdown("### Efficient Frontier and HRP Portfolio Visualization")

    # === Efficient Frontier + CAL ===
    fig, ax = plt.subplots(figsize=(10, 7), facecolor='#121212')

    mu_list = np.array(results_ef[0])
    sigma_list = np.array(results_ef[1])
    sharpe_list = np.array(results_ef[2])

    if len(mu_list) == 0 or len(sigma_list) == 0:
        st.warning("No data to draw efficient frontier.")
        return

    # Efficient Frontier Scatter
    scatter = ax.scatter(
        sigma_list * 100, mu_list * 100, c=sharpe_list,
        cmap='viridis', s=25, alpha=0.8, edgecolors='k', linewidths=0.3
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Sharpe Ratio", color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    # Risk-Free Rate
    ax.scatter(0, rf * 100, c='blue', s=100, marker='o', label=f"Risk-Free Rate ({rf*100:.2f}%)")

    # Optimal Risky Portfolio
    ax.scatter(sigma_p * 100, mu_p * 100, c='red', s=200, marker='*', label="Optimal Risky Portfolio")

    # Capital Allocation Line (CAL)
    cal_x = np.linspace(0, max(sigma_list) * 1.5, 100)
    cal_y = rf + ((mu_p - rf) / sigma_p) * cal_x
    ax.plot(cal_x * 100, cal_y * 100, 'r--', linewidth=2, label="Capital Allocation Line (CAL)")

    # Optimal Complete Portfolio
    ax.scatter(sigma_c * 100, expected_rc * 100, c='lime', s=150, marker='D',
               label=f"Optimal Complete Portfolio (y={y_capped:.2f})")

    # Aesthetic
    ax.set_facecolor('#121212')
    fig.patch.set_facecolor('#121212')
    ax.set_title("Efficient Frontier with Optimal Complete Portfolio", color='white', fontsize=14)
    ax.set_xlabel("Volatility (Risk) (%)", color='white')
    ax.set_ylabel("Expected Return (%)", color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.grid(False)
    ax.legend(facecolor='#1e1e1e', labelcolor='white', fontsize=9)

    st.pyplot(fig)
