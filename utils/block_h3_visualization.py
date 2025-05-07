# utils/block_h3_visualization.py

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def run(hrp_result_dict, benchmark_return_mean, results_ef, best_portfolio,
        mu_p, cov, rf, sigma_c, expected_rc, y_capped, y_opt, tickers, weights):

    st.markdown("### Efficient Frontier with Optimal Complete Portfolio")

    # --- HRP Portfolio Bar Chart ---
    combos = list(hrp_result_dict.keys())[:5]
    returns, vols, cvars, labels = [], [], [], []

    for x in combos:
        key = '-'.join(x) if isinstance(x, tuple) else str(x)
        res = hrp_result_dict.get(x, None) or hrp_result_dict.get(key, None)
        if res:
            returns.append(res['Expected Return (%)'])
            vols.append(res['Volatility (%)'])
            cvars.append(res['CVaR (%)'])
            labels.append(key)

    if not returns:
        st.warning("No valid portfolios available for visualization.")
        return

    x = np.arange(len(labels))
    width = 0.25
    fig1, ax1 = plt.subplots(figsize=(10, 5), facecolor='black')
    ax1.bar(x - width, returns, width, label='Return (%)', color='skyblue')
    ax1.bar(x, vols, width, label='Volatility (%)', color='orange')
    ax1.bar(x + width, cvars, width, label='CVaR (%)', color='salmon')
    ax1.axhline(benchmark_return_mean * 100, color='lime', linestyle='--', linewidth=2,
                label=f"Benchmark Return ({benchmark_return_mean * 100:.2f}%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45)
    ax1.set_facecolor('black')
    ax1.set_title("HRP Portfolio Comparison", color='white')
    ax1.set_ylabel("Percentage (%)", color='white')
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', colors='white')
    ax1.legend(facecolor='black', labelcolor='white')
    st.pyplot(fig1)

    # --- Efficient Frontier ---
    mu_list = np.array(results_ef[0])
    sigma_list = np.array(results_ef[1])
    sharpe_list = np.array(results_ef[2])

    if len(mu_list) == 0 or len(sigma_list) == 0:
        st.warning("Insufficient data for efficient frontier.")
        return

    sigma_p = np.sqrt(np.dot(weights, np.dot(cov, weights)))

    fig2, ax2 = plt.subplots(figsize=(12, 8))
    scatter = ax2.scatter(sigma_list * 100, mu_list * 100, c=sharpe_list, cmap='viridis', alpha=0.5)
    plt.colorbar(scatter, label="Sharpe Ratio")

    ax2.scatter(sigma_p * 100, mu_p * 100, c='red', marker='*', s=200, label='Optimal Risky Portfolio')
    ax2.scatter(0, rf * 100, c='blue', marker='o', s=100, label=f'Risk-Free Rate ({rf * 100:.2f}%)')

    # CAL
    x = np.linspace(0, max(sigma_list) * 1.2, 100)
    y = rf + (mu_p - rf) / sigma_p * x
    ax2.plot(x * 100, y * 100, 'r--', label='Capital Allocation Line (CAL)')

    # Optimal Complete Portfolio
    ax2.scatter(sigma_c * 100, expected_rc * 100, c='green', marker='D', s=150,
                label=f'Optimal Complete Portfolio (y={y_capped:.2f})')

    ax2.set_title("Efficient Frontier with Optimal Complete Portfolio")
    ax2.set_xlabel("Volatility (Risk) (%)")
    ax2.set_ylabel("Expected Return (%)")
    ax2.legend()
    ax2.grid(False)
    plt.tight_layout()
    st.pyplot(fig2)

    st.markdown("#### Selected Tickers")
    st.write(f"{', '.join(tickers)}")
