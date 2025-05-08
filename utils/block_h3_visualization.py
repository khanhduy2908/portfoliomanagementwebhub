import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

def run(hrp_result_dict, benchmark_return_mean, results_ef, best_portfolio,
        mu_p, sigma_p, rf, sigma_c, expected_rc, y_capped, y_opt, tickers, weights, cov):

    st.markdown("### Efficient Frontier and Capital Allocation Line (CAL)")

    fig, ax = plt.subplots(figsize=(10, 7), facecolor='#121212')

    mu_list = np.array(results_ef[0])
    sigma_list = np.array(results_ef[1])
    sharpe_list = np.array(results_ef[2])

    if len(mu_list) == 0 or len(sigma_list) == 0:
        st.warning("No data available to plot the efficient frontier.")
        return

    # Normalize Sharpe ratios for better contrast
    norm = plt.Normalize(min(sharpe_list), max(sharpe_list))

    # Efficient Frontier Scatter with color bar
    scatter = ax.scatter(
        sigma_list, mu_list,
        c=sharpe_list, cmap='plasma', norm=norm,
        s=60, alpha=0.85, edgecolors='white', linewidths=0.3
    )
    cbar = plt.colorbar(scatter, ax=ax, pad=0.01)
    cbar.set_label("Sharpe Ratio", color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    # Risk-Free Rate point
    ax.scatter(0, rf * 100, c='blue', s=130, marker='o', label=f"Risk-Free Rate ({rf*100:.2f}%)")

    # Optimal Risky Portfolio
    ax.scatter(sigma_p * 100, mu_p * 100, c='red', s=180, marker='*', label="Optimal Risky Portfolio")

    # Capital Allocation Line (CAL)
    cal_x = np.linspace(0, max(sigma_list) * 1.4, 100)
    cal_y = rf + ((mu_p - rf) / sigma_p) * cal_x
    ax.plot(cal_x * 100, cal_y * 100, color='red', linewidth=2.5, linestyle='--', label="Capital Allocation Line (CAL)")

    # Optimal Complete Portfolio (after capital allocation)
    ax.scatter(sigma_c * 100, expected_rc * 100, c='lime', s=150, marker='D',
               label=f"Optimal Complete Portfolio (y={y_capped:.2f})")

    # Final visual settings
    ax.set_facecolor('#121212')
    fig.patch.set_facecolor('#121212')
    ax.set_title("Efficient Frontier with Optimal Complete Portfolio", fontsize=14, color='white')
    ax.set_xlabel("Portfolio Volatility (%)", color='white')
    ax.set_ylabel("Expected Return (%)", color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1e1e1e', labelcolor='white', fontsize=9, loc='upper left', frameon=True)
    ax.grid(False)

    st.pyplot(fig)
