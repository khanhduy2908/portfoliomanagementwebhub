
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from sklearn.linear_model import LinearRegression

def evaluate_portfolio(best_portfolio, returns_pivot_stocks, returns_benchmark, rf, A, start_date, end_date):
    num_months = returns_pivot_stocks.shape[0]
    date_range = pd.date_range(start=start_date, periods=num_months, freq='MS')
    returns_pivot_stocks.index = date_range
    returns_benchmark.index = date_range

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

    if len(cumulative_returns) >= 2:
        years = (cumulative_returns.index[-1] - cumulative_returns.index[0]).days / 365.25
        cagr = cumulative_returns.iloc[-1]**(1 / years) - 1 if years > 0 else np.nan
        calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else np.nan
    else:
        cagr = calmar_ratio = np.nan

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

    return summary_df.round(4), benchmark_comparison.round(4), cumulative_returns, benchmark_cumulative, drawdown, rolling_sharpe


def plot_e1_asset_performance_and_risk_bubble(returns_df, tickers, rf_monthly, start_date, end_date):
    df = returns_df[tickers].copy()
    df = df.loc[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
    if df.empty:
        raise ValueError("❌ Không có dữ liệu sau khi lọc thời gian.")

    daily_returns = df / 100
    cum_returns = (1 + daily_returns).cumprod()

    annualized_returns = daily_returns.mean() * 12
    annualized_volatility = daily_returns.std() * np.sqrt(12)
    sharpe_ratios = (annualized_returns - rf_monthly) / annualized_volatility

    fig, ax = plt.subplots(1, 2, figsize=(16, 6), facecolor='white')

    for ticker in tickers:
        ax[0].plot(cum_returns.index, cum_returns[ticker], label=ticker, linewidth=2)
    ax[0].set_title("Historical Cumulative Performance", fontsize=13)
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Cumulative Return")
    ax[0].legend()
    ax[0].grid(False)

    colors = cm.coolwarm((sharpe_ratios - sharpe_ratios.min()) / (sharpe_ratios.max() - sharpe_ratios.min()))
    norm = Normalize(vmin=sharpe_ratios.min(), vmax=sharpe_ratios.max())
    cmap = cm.coolwarm

    for i, ticker in enumerate(tickers):
        color_val = colors[i]
        color_rgb = cmap(norm(sharpe_ratios[ticker]))
        luminance = 0.299 * color_rgb[0] + 0.587 * color_rgb[1] + 0.114 * color_rgb[2]
        text_color = 'black' if luminance > 0.6 else 'white'

        ax[1].scatter(
            annualized_volatility[ticker] * 100,
            annualized_returns[ticker] * 100,
            s=1500,
            c=[color_val],
            edgecolors='black',
            linewidths=1.5,
            alpha=0.95,
            zorder=3
        )
        ax[1].annotate(
            ticker,
            (annualized_volatility[ticker] * 100, annualized_returns[ticker] * 100),
            color=text_color,
            weight='bold',
            ha='center',
            va='center',
            fontsize=10,
            zorder=4
        )

    ax[1].set_title("Risk vs Return (Annualized)", fontsize=13)
    ax[1].set_xlabel("Volatility (%)")
    ax[1].set_ylabel("Return (%)")
    ax[1].grid(False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax[1])
    cbar.set_label('Sharpe Ratio')
    plt.show()
