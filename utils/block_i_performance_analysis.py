import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import config


def run(best_portfolio, returns_pivot_stocks, returns_benchmark,
        rf, A, total_capital,
        data_stocks, data_benchmark, benchmark_symbol,
        weights, tickers_portfolio, start_date, end_date):
    
    print("Block I: Performance Evaluation")

    # --- STEP 1: Return Series Alignment ---
    portfolio_returns_df = returns_pivot_stocks[tickers_portfolio].copy()
    benchmark_returns_df = returns_benchmark.copy()

    common_dates = portfolio_returns_df.index.intersection(benchmark_returns_df.index)
    portfolio_returns_df = portfolio_returns_df.loc[common_dates]
    benchmark_returns_df = benchmark_returns_df.loc[common_dates]
    
    portfolio_returns = portfolio_returns_df @ weights
    benchmark_returns = benchmark_returns_df['Benchmark_Return']

    # --- STEP 2: Cumulative Return ---
    cumulative_portfolio = (1 + portfolio_returns / 100).cumprod()
    cumulative_benchmark = (1 + benchmark_returns / 100).cumprod()

    cumulative_portfolio /= cumulative_portfolio.iloc[0]
    cumulative_benchmark /= cumulative_benchmark.iloc[0]

    # --- STEP 3: Metrics Calculation ---
    mean_return = portfolio_returns.mean()
    volatility = portfolio_returns.std()
    sharpe_ratio = (mean_return - rf * 100) / volatility if volatility != 0 else np.nan

    downside = portfolio_returns[portfolio_returns < rf * 100]
    sortino_ratio = (mean_return - rf * 100) / downside.std() if not downside.empty else np.nan

    drawdown = cumulative_portfolio / cumulative_portfolio.cummax() - 1
    max_drawdown = drawdown.min()

    years = (cumulative_portfolio.index[-1] - cumulative_portfolio.index[0]).days / 365.25
    cagr = cumulative_portfolio.iloc[-1]**(1 / years) - 1 if years > 0 else np.nan
    calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else np.nan

    # --- STEP 4: Alpha / Beta / Tracking Error ---
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

    # --- STEP 5: Rolling Sharpe Ratio ---
    rolling_sharpe = (portfolio_returns - rf * 100).rolling(12).mean() / portfolio_returns.rolling(12).std()

    # --- STEP 6: Summary Table ---
    benchmark_cagr = cumulative_benchmark.iloc[-1]**(1 / years) - 1 if years > 0 else np.nan

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
            cumulative_benchmark.pct_change().min(),
            benchmark_cagr,
            np.nan
        ]
    })

    regression_df = pd.DataFrame({
        'Metric': ['Alpha', 'Beta', 'R-squared', 'Tracking Error', 'Information Ratio'],
        'Value': [alpha, beta, r_squared, tracking_error, info_ratio]
    })

    print("\nPortfolio Performance Summary:")
    print(summary_df.round(4))
    print("\nBenchmark Regression Statistics:")
    print(regression_df.round(4))

    # --- STEP 7: Plots ---
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_portfolio.index, cumulative_portfolio, label='Portfolio', color='blue')
    plt.plot(cumulative_benchmark.index, cumulative_benchmark, label='Benchmark', color='red')
    plt.title("Cumulative Returns (Normalized)")
    plt.xlabel("Date")
    plt.ylabel("Growth Index")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.fill_between(drawdown.index, drawdown, color='red', alpha=0.3)
    plt.title("Portfolio Drawdown")
    plt.ylabel("Drawdown (%)")
    plt.tight_layout()
    plt.grid(False)
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(rolling_sharpe.index, rolling_sharpe, label='12M Rolling Sharpe', color='purple')
    plt.axhline(0, linestyle='--', color='black')
    plt.title("Rolling Sharpe Ratio (12 months)")
    plt.xlabel("Time")
    plt.tight_layout()
    plt.grid(False)
    plt.show()

    return summary_df, regression_df
