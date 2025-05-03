def run(weights, tickers_portfolio, returns_pivot_stocks, returns_benchmark, rf):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression

    portfolio_weights = np.array(weights)
    benchmark_returns = returns_benchmark['Benchmark_Return']
    returns_portfolio_df = returns_pivot_stocks[tickers_portfolio].copy()
    
    # Đồng bộ thời gian
    aligned_index = returns_portfolio_df.index.intersection(benchmark_returns.index)
    returns_portfolio_df = returns_portfolio_df.loc[aligned_index]
    benchmark_returns = benchmark_returns.loc[aligned_index]
    
    # Tính lợi suất danh mục
    portfolio_returns = returns_portfolio_df @ portfolio_weights

    # Tích lũy lợi suất
    cumulative_portfolio = (1 + portfolio_returns / 100).cumprod()
    cumulative_benchmark = (1 + benchmark_returns / 100).cumprod()

    cumulative_portfolio /= cumulative_portfolio.iloc[0]
    cumulative_benchmark /= cumulative_benchmark.iloc[0]

    # Các chỉ số đánh giá hiệu suất
    mean_return = portfolio_returns.mean()
    volatility = portfolio_returns.std()
    sharpe = (mean_return - rf * 100) / volatility if volatility > 0 else np.nan

    downside = portfolio_returns[portfolio_returns < rf * 100]
    sortino = (mean_return - rf * 100) / downside.std() if not downside.empty else np.nan

    drawdown = cumulative_portfolio / cumulative_portfolio.cummax() - 1
    max_drawdown = drawdown.min()

    years = (cumulative_portfolio.index[-1] - cumulative_portfolio.index[0]).days / 365.25
    cagr = cumulative_portfolio.iloc[-1]**(1 / years) - 1 if years > 0 else np.nan
    calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else np.nan

    # Alpha, Beta, Tracking Error, Information Ratio
    aligned_df = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    aligned_df.columns = ['Portfolio', 'Benchmark']
    X = aligned_df['Benchmark'].values.reshape(-1, 1)
    y = aligned_df['Portfolio'].values
    reg = LinearRegression().fit(X, y)
    alpha = reg.intercept_
    beta = reg.coef_[0]
    r_squared = reg.score(X, y)
    tracking_error = np.std(aligned_df['Portfolio'] - aligned_df['Benchmark'])
    info_ratio = (mean_return - benchmark_returns.mean()) / tracking_error if tracking_error > 0 else np.nan

    # Bảng tổng hợp
    summary_df = pd.DataFrame({
        'Metric': ['Mean Return', 'Volatility', 'Sharpe Ratio', 'Sortino Ratio',
                   'Max Drawdown', 'CAGR', 'Calmar Ratio'],
        'Value': [mean_return, volatility, sharpe, sortino,
                  max_drawdown, cagr, calmar]
    })

    regression_df = pd.DataFrame({
        'Metric': ['Alpha', 'Beta', 'R-squared', 'Tracking Error', 'Information Ratio'],
        'Value': [alpha, beta, r_squared, tracking_error, info_ratio]
    })

    # Biểu đồ
    plt.figure(figsize=(12, 5))
    plt.plot(cumulative_portfolio.index, cumulative_portfolio, label='Portfolio')
    plt.plot(cumulative_benchmark.index, cumulative_benchmark, label='Benchmark')
    plt.title("Cumulative Returns (Normalized)")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.fill_between(drawdown.index, drawdown, color='red', alpha=0.3)
    plt.title("Portfolio Drawdown")
    plt.tight_layout()
    plt.grid(False)
    plt.show()

    return summary_df.round(4), regression_df.round(4)
