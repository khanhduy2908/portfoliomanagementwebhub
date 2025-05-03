def run(tickers, returns_pivot_stocks, window_size=12, rf=0.0075):
    import numpy as np
    import pandas as pd
    from scipy.optimize import minimize
    import matplotlib.pyplot as plt

    def max_sharpe(weights, mean_returns, cov_matrix, rf):
        port_return = np.dot(weights, mean_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (port_return - rf*100) / port_vol
        return -sharpe  # Maximize Sharpe

    returns_df = returns_pivot_stocks[tickers].dropna()
    all_dates = returns_df.index
    results = []

    for i in range(window_size, len(all_dates)):
        train_window = returns_df.iloc[i - window_size:i]
        test_month = returns_df.iloc[i]

        mean_ret = train_window.mean()
        cov = train_window.cov()

        x0 = np.ones(len(tickers)) / len(tickers)
        bounds = [(0, 1) for _ in range(len(tickers))]
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

        opt = minimize(max_sharpe, x0, args=(mean_ret, cov, rf), method='SLSQP',
                       bounds=bounds, constraints=constraints)

        if opt.success:
            weights = opt.x
            realized_return = np.dot(weights, test_month)
            results.append({
                'Date': all_dates[i],
                'Return': realized_return,
                'Volatility': np.sqrt(np.dot(weights.T, np.dot(cov, weights))),
                'Sharpe': (realized_return - rf*100) / np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
            })

    backtest_df = pd.DataFrame(results).set_index('Date')

    # Plot
    cum_return = (1 + backtest_df['Return'] / 100).cumprod()
    drawdown = cum_return / cum_return.cummax() - 1

    plt.figure(figsize=(12, 5))
    plt.plot(cum_return.index, cum_return, label='Walkforward Portfolio')
    plt.title("Walkforward Cumulative Return")
    plt.legend()
    plt.tight_layout()
    plt.grid(False)
    plt.show()

    plt.figure(figsize=(12, 3))
    plt.fill_between(drawdown.index, drawdown, color='red', alpha=0.3)
    plt.title("Walkforward Drawdown")
    plt.tight_layout()
    plt.grid(False)
    plt.show()

    return backtest_df.round(4)
