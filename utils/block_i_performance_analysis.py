import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import streamlit as st

plt.style.use('dark_background')

def run(best_portfolio, returns_pivot_stocks, returns_benchmark,
        rf, A, total_capital,
        data_stocks, data_benchmark, benchmark_symbol,
        weights, tickers_portfolio, start_date, end_date):

    # Align date index
    num_months = returns_pivot_stocks.shape[0]
    date_range = pd.date_range(start=start_date, periods=num_months, freq='MS')

    portfolio_returns_df = returns_pivot_stocks[tickers_portfolio].copy()
    benchmark_returns_df = returns_benchmark.copy()

    portfolio_returns_df.index = date_range
    benchmark_returns_df.index = date_range

    portfolio_returns = portfolio_returns_df @ weights
    benchmark_returns = benchmark_returns_df['Benchmark_Return']

    # Cumulative returns
    cumulative_returns = (1 + portfolio_returns / 100).cumprod()
    benchmark_cumulative = (1 + benchmark_returns / 100).cumprod()
    cumulative_returns /= cumulative_returns.iloc[0]
    benchmark_cumulative /= benchmark_cumulative.iloc[0]

    # Metrics
    mean_return = portfolio_returns.mean()
    volatility = portfolio_returns.std()
    sharpe_ratio = (mean_return - rf * 100) / volatility if volatility > 0 else np.nan
    downside = portfolio_returns[portfolio_returns < rf * 100]
    sortino_ratio = (mean_return - rf * 100) / downside.std() if not downside.empty else np.nan
    drawdown = cumulative_returns / cumulative_returns.cummax() - 1
    max_drawdown = drawdown.min()

    years = (cumulative_returns.index[-1] - cumulative_returns.index[0]).days / 365.25
    cagr = cumulative_returns.iloc[-1]**(1 / years) - 1 if years > 0 else np.nan
    calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else np.nan

    # Rolling Sharpe
    rolling_sharpe = (portfolio_returns - rf * 100).rolling(12).mean() / portfolio_returns.rolling(12).std()

    # Alpha/Beta
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

    # Summary tables
    benchmark_cagr = benchmark_cumulative.iloc[-1]**(1 / years) - 1 if years > 0 else np.nan
    summary_df = pd.DataFrame({
        'Metric': ['Mean Return', 'Volatility', 'Sharpe Ratio', 'Sortino Ratio',
                   'Max Drawdown', 'CAGR', 'Calmar Ratio'],
        'Portfolio': [mean_return, volatility, sharpe_ratio, sortino_ratio,
                      max_drawdown, cagr, calmar_ratio],
        'Benchmark': [
            benchmark_returns.mean(),
            benchmark_returns.std(),
            (benchmark_returns.mean() - rf * 100) / benchmark_returns.std() if benchmark_returns.std() > 0 else np.nan,
            np.nan,
            benchmark_cumulative.pct_change().min(),
            benchmark_cagr,
            np.nan
        ]
    })

    regression_df = pd.DataFrame({
        'Metric': ['Alpha', 'Beta', 'R-squared', 'Tracking Error', 'Information Ratio'],
        'Value': [alpha, beta, r_squared, tracking_error, info_ratio]
    })

    # Layout: Row 1
    st.subheader("Cumulative Returns")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(cumulative_returns.index, cumulative_returns, label='Portfolio', color='cyan')
    ax1.plot(benchmark_cumulative.index, benchmark_cumulative, label='Benchmark', color='magenta')
    ax1.set_title("Cumulative Returns (Normalized)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Index")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)

    # Layout: Row 2 with 2 charts side-by-side
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Drawdown")
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        ax2.fill_between(drawdown.index, drawdown, color='red', alpha=0.4)
        ax2.set_title("Drawdown")
        ax2.set_ylabel("Drawdown (%)")
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

    with col2:
        st.subheader("12M Rolling Sharpe Ratio")
        fig3, ax3 = plt.subplots(figsize=(6, 3))
        ax3.plot(rolling_sharpe.index, rolling_sharpe, color='orange')
        ax3.axhline(0, linestyle='--', color='white', alpha=0.4)
        ax3.set_title("12-Month Rolling Sharpe")
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3)

    # Final tables
    st.subheader("Performance Summary")
    st.dataframe(summary_df.round(4), use_container_width=True)

    st.subheader("Regression Statistics")
    st.dataframe(regression_df.round(4), use_container_width=True)

    return summary_df, regression_df
