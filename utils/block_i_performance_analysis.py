# --- BLOCK I: Comprehensive Performance Evaluation (9 Institutional Charts) ---
# File: utils/block_i_performance.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from sklearn.linear_model import LinearRegression

def run(best_portfolio, returns_pivot_stocks, returns_benchmark, 
        data_stocks, data_benchmark, benchmark_symbol, 
        rf, A, start_date, end_date, 
        hrp_cvar_results, adj_returns_combinations, cov_matrix_dict):

    # --- Portfolio returns ---
    weights = np.array(list(best_portfolio['Weights'].values()))
    tickers = list(best_portfolio['Weights'].keys())

    returns_portfolio_df = returns_pivot_stocks[tickers].copy()
    benchmark_returns_df = returns_benchmark.copy()

    returns_portfolio_df.index = pd.to_datetime(returns_portfolio_df.index)
    benchmark_returns_df.index = pd.to_datetime(benchmark_returns_df.index)
    common_dates = returns_portfolio_df.index.intersection(benchmark_returns_df.index)
    returns_portfolio_df = returns_portfolio_df.loc[common_dates]
    benchmark_returns_df = benchmark_returns_df.loc[common_dates]

    portfolio_returns = returns_portfolio_df @ weights
    benchmark_returns = benchmark_returns_df['Benchmark_Return']

    cumulative_returns = (1 + portfolio_returns / 100).cumprod()
    benchmark_cumulative = (1 + benchmark_returns / 100).cumprod()
    cumulative_returns /= cumulative_returns.iloc[0]
    benchmark_cumulative /= benchmark_cumulative.iloc[0]

    mean_return = portfolio_returns.mean()
    volatility = portfolio_returns.std()
    sharpe_ratio = (mean_return - rf * 100) / volatility if volatility != 0 else np.nan
    downside = portfolio_returns[portfolio_returns < rf * 100]
    sortino_ratio = (mean_return - rf * 100) / downside.std() if not downside.empty else np.nan

    drawdown = cumulative_returns / cumulative_returns.cummax() - 1
    max_drawdown = drawdown.min()

    years = (cumulative_returns.index[-1] - cumulative_returns.index[0]).days / 365.25
    cagr = cumulative_returns.iloc[-1]**(1 / years) - 1 if years > 0 else np.nan
    calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else np.nan

    rolling_sharpe = (portfolio_returns - rf * 100).rolling(12).mean() / portfolio_returns.rolling(12).std()

    aligned_df = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    aligned_df.columns = ['Portfolio', 'Benchmark']
    if len(aligned_df) >= 2:
        X = aligned_df['Benchmark'].values.reshape(-1, 1)
        y = aligned_df['Portfolio'].values
        reg = LinearRegression().fit(X, y)
        alpha = reg.intercept_
        beta = reg.coef_[0]
        r_squared = reg.score(X, y)
        tracking_error = np.std(aligned_df['Portfolio'] - aligned_df['Benchmark'])
        info_ratio = (mean_return - benchmark_returns.mean()) / tracking_error if tracking_error != 0 else np.nan
    else:
        alpha = beta = r_squared = tracking_error = info_ratio = np.nan

    benchmark_cagr = benchmark_cumulative.iloc[-1]**(1 / years) - 1 if years > 0 else np.nan

    # --- Summary Tables ---
    summary_df = pd.DataFrame({
        'Metric': ['Mean Return', 'Volatility', 'Sharpe Ratio', 'Sortino Ratio',
                   'Max Drawdown', 'CAGR', 'Calmar Ratio'],
        'Portfolio': [mean_return, volatility, sharpe_ratio, sortino_ratio,
                      max_drawdown, cagr, calmar_ratio],
        'Benchmark': [
            benchmark_returns.mean(),
            benchmark_returns.std(),
            (benchmark_returns.mean() - rf * 100) / benchmark_returns.std() if benchmark_returns.std() != 0 else np.nan,
            np.nan,
            benchmark_cumulative.pct_change().min(),
            benchmark_cagr,
            np.nan
        ]
    })

    benchmark_comparison = pd.DataFrame({
        'Metric': ['Alpha', 'Beta', 'R-squared', 'Tracking Error', 'Information Ratio'],
        'Value': [alpha, beta, r_squared, tracking_error, info_ratio]
    })

    print("\nPortfolio Performance Summary:")
    print(summary_df.round(4))
    print("\nBenchmark Regression Statistics:")
    print(benchmark_comparison.round(4))

    # --- Chart 1: Cumulative Returns ---
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_returns, label='Portfolio', color='blue')
    plt.plot(benchmark_cumulative, label='Benchmark', color='red')
    plt.title("Cumulative Returns (Normalized)")
    plt.xlabel("Date")
    plt.ylabel("Index Value")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    # --- Chart 2: Portfolio Drawdown ---
    plt.figure(figsize=(14, 4))
    plt.fill_between(drawdown.index, drawdown, color='red', alpha=0.3)
    plt.title("Portfolio Drawdown")
    plt.ylabel("Drawdown (%)")
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    # --- Chart 3: 12M Rolling Sharpe Ratio ---
    plt.figure(figsize=(14, 4))
    plt.plot(rolling_sharpe.index, rolling_sharpe, color='purple')
    plt.axhline(0, linestyle='--', color='black')
    plt.title("12M Rolling Sharpe Ratio")
    plt.xlabel("Date")
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    # --- Chart 4–5: Asset Risk/Return & Cumulative Performance (subplot) ---
    daily_returns = returns_pivot_stocks[tickers] / 100
    cum_returns = (1 + daily_returns).cumprod()

    ann_returns = daily_returns.mean() * 12
    ann_vols = daily_returns.std() * np.sqrt(12)
    sharpe_assets = (ann_returns - rf) / ann_vols

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    for ticker in tickers:
        ax[0].plot(cum_returns[ticker], label=ticker)
    ax[0].set_title("Cumulative Performance by Asset")
    ax[0].legend()
    ax[0].grid(False)

    norm = Normalize(vmin=sharpe_assets.min(), vmax=sharpe_assets.max())
    colors = cm.coolwarm(norm(sharpe_assets))

    for i, ticker in enumerate(tickers):
        ax[1].scatter(ann_vols[ticker]*100, ann_returns[ticker]*100,
                      s=1000, c=[colors[i]], edgecolors='black')
        ax[1].annotate(ticker, (ann_vols[ticker]*100, ann_returns[ticker]*100),
                       ha='center', va='center', fontsize=9, fontweight='bold')

    ax[1].set_title("Risk vs Return Bubble Chart")
    ax[1].set_xlabel("Volatility (%)")
    ax[1].set_ylabel("Return (%)")
    ax[1].grid(False)

    plt.tight_layout()
    plt.show()

    # --- Chart 6–7: Portfolio vs Benchmark (subplot) ---
    df_stocks = data_stocks[data_stocks['Ticker'].isin(tickers)].copy()
    df_bench = data_benchmark[data_benchmark['Ticker'] == benchmark_symbol].copy()

    df_stocks['time'] = pd.to_datetime(df_stocks['time'])
    df_bench['time'] = pd.to_datetime(df_bench['time'])

    prices = df_stocks.pivot(index='time', columns='Ticker', values='Close').sort_index()
    bench_price = df_bench.pivot(index='time', columns='Ticker', values='Close').sort_index()
    bench_price = bench_price[[benchmark_symbol]]

    common = prices.index.intersection(bench_price.index)
    prices = prices.loc[common]
    bench_price = bench_price.loc[common]

    port_ret = prices.pct_change().dropna() @ weights
    bench_ret = bench_price[benchmark_symbol].pct_change().dropna()

    cum_p = (1 + port_ret).cumprod()
    cum_b = (1 + bench_ret).cumprod()

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    ax[0].plot(cum_p * 100, label='Portfolio', color='dodgerblue')
    ax[0].plot(cum_b * 100, label=benchmark_symbol, color='orangered')
    ax[0].set_title("Cumulative Returns vs Benchmark")
    ax[0].legend()
    ax[0].grid(False)

    for i, (ret, vol, label) in enumerate(zip(
        [cum_p.iloc[-1], cum_b.iloc[-1]],
        [port_ret.std() * np.sqrt(12), bench_ret.std() * np.sqrt(12)],
        ['Portfolio', 'Benchmark']
    )):
        ax[1].scatter(vol * 100, ret * 100, s=1500, c='skyblue', edgecolors='black')
        ax[1].annotate(label, (vol * 100, ret * 100), ha='center', va='center')

    ax[1].set_title("Risk-Return Bubble")
    ax[1].set_xlabel("Volatility (%)")
    ax[1].set_ylabel("Cumulative Return (%)")
    ax[1].grid(False)

    plt.tight_layout()
    plt.show()

    # --- Chart 8–9: Efficient Frontier + CAL + Optimal Points ---
    mu = np.array([adj_returns_combinations[best_portfolio['Portfolio']][t] for t in tickers]) / 100
    cov = cov_matrix_dict[best_portfolio['Portfolio']].loc[tickers, tickers].values

    def perf(w): return np.dot(w, mu), np.sqrt(w.T @ cov @ w)
    n = 5000
    results = np.zeros((3, n))
    for i in range(n):
        w = np.random.random(len(tickers))
        w /= np.sum(w)
        r, v = perf(w)
        results[:, i] = [r, v, (r - rf) / v if v > 0 else 0]

    mu_p = mu @ weights
    sigma_p = np.sqrt(weights.T @ cov @ weights)
    y_opt = (mu_p - rf) / (A * sigma_p**2)
    y_capped = max(0.6, min(y_opt, 0.9))
    E_rc = y_capped * mu_p + (1 - y_capped) * rf
    sigma_c = y_capped * sigma_p

    plt.figure(figsize=(12, 8))
    sc = plt.scatter(results[1, :]*100, results[0, :]*100, c=results[2, :], cmap='viridis', alpha=0.4)
    plt.colorbar(sc, label="Sharpe Ratio")
    plt.scatter(sigma_p*100, mu_p*100, c='red', marker='*', s=200, label='Optimal Risky Portfolio')
    plt.scatter(0, rf*100, c='blue', marker='o', s=100, label='Risk-Free Rate')

    # CAL
    x = np.linspace(0, max(results[1, :])*1.2, 100)
    y = rf + (mu_p - rf) / sigma_p * x
    plt.plot(x*100, y*100, 'r--', label='CAL')

    plt.scatter(sigma_c*100, E_rc*100, c='green', marker='D', s=150,
                label=f'Optimal Complete Portfolio (y={y_capped:.2f})')

    plt.title("Efficient Frontier & Capital Allocation Line")
    plt.xlabel("Volatility (%)")
    plt.ylabel("Expected Return (%)")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()
