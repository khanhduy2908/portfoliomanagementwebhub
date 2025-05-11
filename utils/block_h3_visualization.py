import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

def run(hrp_result_dict, benchmark_return_mean, results_ef, best_portfolio,
        mu_p, sigma_p, rf, sigma_c, expected_rc, y_capped, y_opt,
        tickers, weights, cov):

    st.markdown("### Efficient Frontier and Capital Allocation Line (CAL)")

    # === 1. Chuẩn bị dữ liệu ===
    mu_list = np.array(results_ef[0])
    sigma_list = np.array(results_ef[1])
    sharpe_list = np.array(results_ef[2])

    # Clean NaN or inf
    mask = ~np.isnan(sharpe_list) & ~np.isinf(sharpe_list)
    mu_list = mu_list[mask]
    sigma_list = sigma_list[mask]
    sharpe_list = sharpe_list[mask]

    if len(mu_list) == 0 or len(sigma_list) == 0:
        st.warning("⚠️ No data available to plot Efficient Frontier.")
        return

    # === 2. Vẽ biểu đồ ===
    fig, ax = plt.subplots(figsize=(10, 7), facecolor="#121212")

    # Efficient Frontier with Gradient Sharpe Ratio
    scatter = ax.scatter(
        sigma_list,
        mu_list,
        c=sharpe_list,
        cmap='plasma',
        s=50,
        edgecolors='black',
        linewidths=0.3,
        alpha=0.9,
        vmin=np.nanmin(sharpe_list),
        vmax=np.nanmax(sharpe_list)
    )

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Sharpe Ratio", color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.get_yticklabels(), color='white')

    # === 3. Risk-Free Rate Point ===
    ax.scatter(0, rf * 100, c='blue', s=100, marker='o', label=f"Risk-Free Rate ({rf*100:.2f}%)")

    # === 4. Optimal Risky Portfolio ===
    ax.scatter(sigma_p * 100, mu_p * 100, c='red', marker='*', s=200, label="Optimal Risky Portfolio")

    # === 5. Capital Allocation Line (CAL) ===
    slope = (mu_p - rf) / sigma_p
    x_cal = np.linspace(0, max(sigma_list.max(), sigma_p * 1.4), 100)
    y_cal = rf + slope * x_cal
    ax.plot(x_cal * 100, y_cal * 100, 'r--', linewidth=2, label="Capital Allocation Line (CAL)")

    # === 6. Complete Portfolio (capped y) ===
    ax.scatter(sigma_c * 100, expected_rc * 100, c='lime', marker='D', s=150,
               label=f"Complete Portfolio (y = {y_capped:.2f})")

    # === 7. Leveraged Portfolio nếu có y_opt > y_capped ===
    if abs(y_opt - y_capped) > 1e-3:
        sigma_uncapped = y_opt * sigma_p
        expected_uncapped = y_opt * mu_p + (1 - y_opt) * rf
        ax.scatter(sigma_uncapped * 100, expected_uncapped * 100,
                   c='magenta', marker='X', s=150,
                   label=f"Leveraged Portfolio (y = {y_opt:.2f})")

    # === 8. Aesthetic setup ===
    ax.set_facecolor('#121212')
    fig.patch.set_facecolor('#121212')
    ax.set_title("Efficient Frontier with CAL and Optimal Portfolios", color='white', fontsize=14)
    ax.set_xlabel("Volatility (%)", color='white')
    ax.set_ylabel("Expected Return (%)", color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1e1e1e', labelcolor='white', fontsize=9, loc='upper left')
    ax.grid(False)

    st.pyplot(fig)
