import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize, to_rgb
from sklearn.linear_model import LinearRegression

def run(best_portfolio, returns_pivot_stocks, returns_benchmark, data_stocks, data_benchmark,
        benchmark_symbol, rf, A, start_date, end_date, hrp_cvar_results,
        adj_returns_combinations, cov_matrix_dict):

    weights = np.array(list(best_portfolio['Weights'].values()))
    tickers_portfolio = list(best_portfolio['Weights'].keys())

    returns_portfolio_df = returns_pivot_stocks[tickers_portfolio].copy()
    benchmark_returns_df = returns_benchmark.copy()

    returns_portfolio_df.index = pd.to_datetime(returns_portfolio_df.index)
    benchmark_returns_df.index = pd.to_datetime(benchmark_returns_df.index)

    common_dates = returns_portfolio_df.index.intersection(benchmark_returns_df.index)
    returns_portfolio_df = returns_portfolio_df.loc[common_dates]
    benchmark_returns_df = benchmark_returns_df.loc[common_dates]

    benchmark_returns = benchmark_returns_df['Benchmark_Return']
    portfolio_returns = returns_portfolio_df @ weights

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

    # 1. Cumulative Returns
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_returns.index, cumulative_returns, label='Portfolio', color='blue')
    plt.plot(benchmark_cumulative.index, benchmark_cumulative, label='Benchmark', color='red')
    plt.title("Cumulative Returns (Normalized)")
    plt.xlabel("Time")
    plt.ylabel("Index Level")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    # 2. Drawdown
    plt.figure(figsize=(14, 4))
    plt.fill_between(drawdown.index, drawdown, color='red', alpha=0.3)
    plt.title("Portfolio Drawdown")
    plt.ylabel("Drawdown (%)")
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    # 3. Rolling Sharpe
    plt.figure(figsize=(14, 4))
    plt.plot(rolling_sharpe.index, rolling_sharpe, label='12M Rolling Sharpe', color='purple')
    plt.axhline(0, linestyle='--', color='black')
    plt.title("12M Rolling Sharpe Ratio")
    plt.xlabel("Time")
    plt.grid(False)
    plt.tight_layout()
    plt.show()
