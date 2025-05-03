import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def run_block_i(best_portfolio, returns_pivot_stocks, returns_benchmark, rf, start_date, end_date):
    tickers_portfolio = list(best_portfolio['Weights'].keys())
    weights = np.array(list(best_portfolio['Weights'].values()))

    returns_portfolio_df = returns_pivot_stocks[tickers_portfolio].copy()
    benchmark_returns_df = returns_benchmark.copy()

    # Align dates
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

    # Metrics
    mean_return = portfolio_returns.mean()
    volatility = portfolio_returns.std()
    sharpe_ratio = (mean_return - rf * 100) / volatility if volatility != 0 else np.nan

    downside = portfolio_returns[portfolio_returns < rf * 100]
    sortino_ratio = (mean_return - rf * 100) / downside.std() if not downside.empty else np.nan

    drawdown = cumulative_returns / cumulative_returns.cummax() - 1
    max_drawdown = drawdown.min()

    if len(cumulative_returns) >= 2:
        years = (cumulative_returns.index[-1] - cumulative_returns.index[0]).days / 365.25
        cagr = cumulative_returns.iloc[-1]**(1 / years) - 1 if years > 0 else np.nan
        calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else np.nan
    else:
        cagr = calmar_ratio = np.nan

    # Alpha / Beta / Info Ratio
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
            benchmark_cumulative.iloc[-1]**(1 / years) - 1 if years > 0 else np.nan,
            np.nan
        ]
    })

    benchmark_comparison = pd.DataFrame({
        'Metric': ['Alpha', 'Beta', 'R-squared', 'Tracking Error', 'Information Ratio'],
        'Value': [alpha, beta, r_squared, tracking_error, info_ratio]
    })

    results = {
        "summary_df": summary_df.round(4),
        "benchmark_comparison": benchmark_comparison.round(4),
        "portfolio_returns": portfolio_returns,
        "benchmark_returns": benchmark_returns,
        "cumulative_returns": cumulative_returns,
        "benchmark_cumulative": benchmark_cumulative,
        "drawdown": drawdown
    }

    return results