def run_block_h(hrp_cvar_results, adj_returns_combinations, cov_matrix_dict,
                rf, A, total_capital, benchmark_return_mean):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import cvxpy as cp

    if not hrp_cvar_results:
        print("No valid portfolio found in HRP-CVaR results.")
        return

    # Select best portfolio
    best_portfolio = max(hrp_cvar_results, key=lambda x: x['Sharpe Ratio'])
    tickers = list(best_portfolio['Weights'].keys())
    weights_hrp = np.array(list(best_portfolio['Weights'].values()))
    mu = np.array([adj_returns_combinations[best_portfolio['Portfolio']][t] for t in tickers]) / 100
    cov = cov_matrix_dict[best_portfolio['Portfolio']].loc[tickers, tickers].values

    # Portfolio stats
    mu_p = float(mu @ weights_hrp)
    sigma_p = np.sqrt(weights_hrp.T @ cov @ weights_hrp)

    y_opt = (mu_p - rf) / (A * sigma_p ** 2)
    y_capped = max(0.6, min(y_opt, 0.9))

    expected_rc = y_capped * mu_p + (1 - y_capped) * rf
    sigma_c = y_capped * sigma_p
    utility = expected_rc - 0.5 * A * sigma_c ** 2

    capital_risky = y_capped * total_capital
    capital_rf = total_capital - capital_risky
    capital_alloc = {t: round(capital_risky * w) for t, w in zip(tickers, weights_hrp)}

    # --- Summary Table ---
    summary_info = {
        "Risk Aversion (A)": A,
        "Expected Return (E_rc)": round(expected_rc, 4),
        "Portfolio Risk (σ_c)": round(sigma_c * 100, 4),
        "Utility (U)": round(utility, 4),
        "Capital (Risk-Free)": round(capital_rf),
        "Capital (Risky)": round(capital_risky)
    }
    summary_df = pd.DataFrame(summary_info.items(), columns=["Metric", "Value"])

    print("\n--- Portfolio Allocation Summary ---")
    print(summary_df.to_string(index=False))

    print("\n--- Capital Allocation Breakdown ---")
    for t, v in capital_alloc.items():
        print(f"{t}: {v:,.0f} VND")

    # --- Pie Chart ---
    labels = ['Risk-Free Asset'] + tickers
    sizes = [capital_rf] + [capital_alloc[t] for t in tickers]
    if np.any(np.array(sizes) < 0):
        print("Warning: Negative capital allocations detected.")
    else:
        plt.figure(figsize=(8, 6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
                shadow=False, textprops={'fontsize': 12})
        plt.title("Optimal Complete Portfolio Allocation")
        plt.tight_layout()
        plt.show()

    # --- Bar Chart Comparison with Benchmark ---
    hrp_result_dict = {p['Portfolio']: p for p in hrp_cvar_results}
    top_n = 5
    top_portfolios = sorted(hrp_result_dict.items(), key=lambda x: x[1]['Sharpe Ratio'], reverse=True)[:top_n]

    combos = [x[0] for x in top_portfolios]
    returns = [x[1]['Expected Return (%)'] for x in top_portfolios]
    vols = [x[1]['Volatility (%)'] for x in top_portfolios]
    cvars = [x[1]['CVaR (%)'] for x in top_portfolios]

    x = np.arange(len(combos))
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, returns, width, label='Return (%)', color='#1f77b4')
    plt.bar(x, vols, width, label='Volatility (%)', color='#ff7f0e')
    plt.bar(x + width, cvars, width, label='CVaR (%)', color='#d62728')
    plt.axhline(benchmark_return_mean * 100, color='green', linestyle='--', linewidth=2,
                label=f"Benchmark Return ({benchmark_return_mean * 100:.2f}%)")

    plt.xticks(x, combos, rotation=45)
    plt.ylabel("Percentage (%)")
    plt.title("Top HRP Portfolios vs Benchmark")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    return summary_df, capital_alloc
