# utils/block_i_performance_analysis.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import streamlit as st

def run(best_portfolio, returns_pivot_stocks, returns_benchmark,
        rf, A, total_capital, data_stocks, data_benchmark,
        benchmark_symbol, weights, tickers_portfolio,
        start_date, end_date):

    st.markdown("### 📊 Portfolio Performance Analysis")

    # --- STEP 1: Prepare Returns ---
    returns_portfolio_df = returns_pivot_stocks[tickers_portfolio].copy()
    benchmark_returns_df = returns_benchmark.copy()

    returns_portfolio_df.index = pd.to_datetime(returns_portfolio_df.index)
    benchmark_returns_df.index = pd.to_datetime(benchmark_returns_df.index)

    common_dates = returns_portfolio_df.index.intersection(benchmark_returns_df.index)
    returns_portfolio_df = returns_portfolio_df.loc[common_dates]
    benchmark_returns_df = benchmark_returns_df.loc[common_dates]

    benchmark_returns = benchmark_returns_df['Benchmark_Return']
    portfolio_returns = returns_portfolio_df @ weights

    # --- STEP 2: Accumulated Return ---
    cum_portfolio = (1 + portfolio_returns / 100).cumprod()
    cum_benchmark = (1 + benchmark_returns / 100).cumprod()
    cum_portfolio /= cum_portfolio.iloc[0]
    cum_benchmark /= cum_benchmark.iloc[0]

    # --- STEP 3: Metrics ---
    mean_return = portfolio_returns.mean()
    volatility = portfolio_returns.std()
    sharpe_ratio = (mean_return - rf * 100) / volatility if volatility > 0 else np.nan

    downside = portfolio_returns[portfolio_returns < rf * 100]
    sortino_ratio = (mean_return - rf * 100) / downside.std() if not downside.empty else np.nan

    drawdown = cum_portfolio / cum_portfolio.cummax() - 1
    max_drawdown = drawdown.min()

    if len(cum_portfolio) >= 2:
        years = (cum_portfolio.index[-1] - cum_portfolio.index[0]).days / 365.25
        cagr = cum_portfolio.iloc[-1]**(1 / years) - 1 if years > 0 else np.nan
        calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else np.nan
    else:
        cagr = calmar_ratio = np.nan

    # --- STEP 4: Alpha, Beta, TE ---
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
        info_ratio = (mean_return - benchmark_returns.mean()) / tracking_error if tracking_error > 0 else np.nan
    else:
        alpha = beta = r_squared = tracking_error = info_ratio = np.nan

    # --- STEP 5: Summary ---
    benchmark_cagr = cum_benchmark.iloc[-1]**(1 / years) - 1 if years > 0 else np.nan
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
            cum_benchmark.pct_change().min(),
            benchmark_cagr,
            np.nan
        ]
    })

    regression_stats = pd.DataFrame({
        'Metric': ['Alpha', 'Beta', 'R-squared', 'Tracking Error', 'Information Ratio'],
        'Value': [alpha, beta, r_squared, tracking_error, info_ratio]
    })

    # --- STEP 6: Show Tables ---
    st.subheader("📈 Performance Metrics")
    st.dataframe(summary_df.round(4), use_container_width=True)

    st.subheader("📉 Benchmark Regression Statistics")
    st.dataframe(regression_stats.round(4), use_container_width=True)

    # --- STEP 7: Cumulative Return ---
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(cum_portfolio.index, cum_portfolio, label='Portfolio', color='navy')
    ax1.plot(cum_benchmark.index, cum_benchmark, label='Benchmark', color='darkred')
    ax1.set_title("Cumulative Return (Normalized)", fontsize=14)
    ax1.legend()
    ax1.grid(False)
    st.pyplot(fig1)

    # --- STEP 8: Drawdown ---
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.fill_between(drawdown.index, drawdown, color='red', alpha=0.3)
    ax2.set_title("Portfolio Drawdown")
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(False)
    st.pyplot(fig2)

    # --- STEP 9: Rolling Sharpe Ratio (12M) ---
    rolling_sharpe = (portfolio_returns - rf * 100).rolling(12).mean() / portfolio_returns.rolling(12).std()
    fig3, ax3 = plt.subplots(figsize=(14, 4))
    ax3.plot(rolling_sharpe.index, rolling_sharpe, label='12M Rolling Sharpe', color='purple')
    ax3.axhline(0, linestyle='--', color='black')
    ax3.set_title("12-Month Rolling Sharpe Ratio")
    ax3.set_xlabel("Time")
    ax3.grid(False)
    st.pyplot(fig3)
