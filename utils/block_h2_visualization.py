
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def run(hrp_result_dict, benchmark_return_mean, results_ef, best_portfolio, mu_p, sigma_p, rf, sigma_c, expected_rc, y_capped, y_opt):
    st.markdown("### HRP vs Benchmark and Efficient Frontier with CAL")

    # --- HRP vs Benchmark (Bar Chart) ---
    st.markdown("#### HRP Portfolios vs Benchmark")
    combos = list(hrp_result_dict.keys())[:5]
    returns = [hrp_result_dict[x]['Expected Return (%)'] for x in combos]
    vols = [hrp_result_dict[x]['Volatility (%)'] for x in combos]
    cvars = [hrp_result_dict[x]['CVaR (%)'] for x in combos]

    x = np.arange(len(combos))
    width = 0.25

    fig1, ax1 = plt.subplots(figsize=(10, 5), facecolor='black')
    ax1.bar(x - width, returns, width, label='Return (%)', color='skyblue')
    ax1.bar(x, vols, width, label='Volatility (%)', color='orange')
    ax1.bar(x + width, cvars, width, label='CVaR (%)', color='salmon')
    ax1.axhline(benchmark_return_mean * 100, color='lime', linestyle='--', linewidth=2,
                label=f"Benchmark Return ({benchmark_return_mean * 100:.2f}%)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(combos, rotation=45)
    ax1.set_facecolor('black')
    ax1.set_title("HRP Portfolio Comparison", color='white')
    ax1.set_ylabel("Percentage (%)", color='white')
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', colors='white')
    ax1.legend(facecolor='black', labelcolor='white')
    st.pyplot(fig1)

    # --- Efficient Frontier + CAL (Scatter Plot) ---
    st.markdown("#### Efficient Frontier and Capital Allocation Line (CAL)")
    fig2, ax2 = plt.subplots(figsize=(10, 5), facecolor='black')
    scatter = ax2.scatter(results_ef[1], results_ef[0], c=results_ef[2], cmap='viridis', alpha=0.6, label='Portfolios')
    plt.colorbar(scatter, ax=ax2, label='Sharpe Ratio')

    ax2.scatter(sigma_p * 100, mu_p * 100, c='red', marker='*', s=200,
                label=f'Optimal Risky Portfolio ({best_portfolio["Portfolio"]})')
    ax2.scatter(0, rf * 100, c='white', marker='o', s=100, label=f'Risk-Free Rate ({rf * 100:.2f}%)')

    slope = (mu_p - rf) / sigma_p
    x_cal = np.linspace(0, max(results_ef[1]) * 1.2, 100)
    y_cal = rf * 100 + slope * x_cal
    ax2.plot(x_cal, y_cal, 'r--', label='Capital Allocation Line (CAL)')
    ax2.scatter(sigma_c * 100, expected_rc * 100, c='lime', marker='D', s=150,
                label=f'Optimal Complete Portfolio (y={y_capped:.2f})')

    if abs(y_opt - y_capped) > 1e-3:
        sigma_uncapped = y_opt * sigma_p
        expected_uncapped = y_opt * mu_p + (1 - y_opt) * rf
        ax2.scatter(sigma_uncapped * 100, expected_uncapped * 100, c='purple', marker='D', s=150,
                    label=f'Optimal Complete Portfolio (y={y_opt:.2f})')

    ax2.set_facecolor('black')
    ax2.set_title('Efficient Frontier with CAL', color='white')
    ax2.set_xlabel('Volatility (%)', color='white')
    ax2.set_ylabel('Expected Return (%)', color='white')
    ax2.tick_params(axis='x', colors='white')
    ax2.tick_params(axis='y', colors='white')
    ax2.legend(facecolor='black', labelcolor='white')
    ax2.grid(False)
    st.pyplot(fig2)
