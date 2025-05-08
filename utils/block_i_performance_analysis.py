import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import streamlit as st

def run(best_portfolio, returns_pivot_stocks, returns_benchmark,
        rf, A, total_capital,
        data_stocks, data_benchmark, benchmark_symbol,
        weights, tickers_portfolio, start_date, end_date):

    # --- Re-align index ---
    returns_pivot_stocks.index = pd.to_datetime(returns_pivot_stocks.index)
    returns_benchmark.index = pd.to_datetime(returns_benchmark.index)
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    common_index = date_range.intersection(returns_pivot_stocks.index)

    if common_index.empty:
        raise ValueError("❌ No return data available for the selected date range.")

    returns_pivot_stocks = returns_pivot_stocks.loc[common_index]
    returns_benchmark = returns_benchmark.loc[common_index]

    # --- Portfolio returns ---
    portfolio_returns = returns_pivot_stocks[tickers_portfolio] @ weights
    benchmark_returns = returns_benchmark['Benchmark_Return']

    # --- Cumulative returns ---
    cumulative_returns = (1 + portfolio_returns / 100).cumprod()
    benchmark_cumulative = (1 + benchmark_returns / 100).cumprod()
    cumulative_returns /= cumulative_returns.iloc[0]
    benchmark_cumulative /= benchmark_cumulative.iloc[0]

    # --- Performance metrics ---
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

    # --- Rolling Sharpe Ratio ---
    rolling_sharpe = (portfolio_returns - rf * 100).rolling(12).mean() / portfolio_returns.rolling(12).std()
    rolling_sharpe = rolling_sharpe.dropna()

    # --- Summary table ---
    benchmark_cagr = benchmark_cumulative.iloc[-1]**(1 / years) - 1 if years > 0 else np.nan
    summary_df = pd.DataFrame({
        'Metric': ['Mean Return (%)', 'Volatility (%)', 'Sharpe Ratio', 'Sortino Ratio',
                   'Max Drawdown (%)', 'CAGR (%)', 'Calmar Ratio'],
        'Portfolio': [mean_return, volatility, sharpe_ratio, sortino_ratio,
                      max_drawdown * 100, cagr * 100, calmar_ratio],
        'Benchmark': [
            benchmark_returns.mean(),
            benchmark_returns.std(),
            (benchmark_returns.mean() - rf * 100) / benchmark_returns.std() if benchmark_returns.std() > 0 else np.nan,
            np.nan,
            (benchmark_cumulative / benchmark_cumulative.cummax() - 1).min() * 100,
            benchmark_cagr * 100,
            np.nan
        ]
    })

    # --- Visualizations ---
    st.subheader("Cumulative Returns")
    fig1, ax1 = plt.subplots(figsize=(10, 4), facecolor='#1e1e1e')
    ax1.plot(cumulative_returns.index, cumulative_returns, label='Portfolio', color='cyan')
    ax1.plot(benchmark_cumulative.index, benchmark_cumulative, label='Benchmark', color='magenta')
    ax1.set_title("Cumulative Returns (Normalized)", color='white')
    ax1.set_xlabel("Date", color='white')
    ax1.set_ylabel("Index Level", color='white')
    ax1.tick_params(colors='white')
    ax1.legend(facecolor='#1e1e1e', labelcolor='white')
    ax1.grid(False)
    fig1.patch.set_facecolor('#1e1e1e')
    st.pyplot(fig1)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Drawdown (%)")
        drawdown.index = pd.to_datetime(drawdown.index)
        fig2, ax2 = plt.subplots(figsize=(6, 3), facecolor='#1e1e1e')
        ax2.fill_between(drawdown.index, drawdown.values * 100, color='red', alpha=0.4)
        ax2.set_title("Portfolio Drawdown", color='white')
        ax2.set_ylabel("Drawdown (%)", color='white')
        ax2.tick_params(colors='white')
        ax2.grid(False)
        ax2.set_facecolor('#1e1e1e')
        fig2.patch.set_facecolor('#1e1e1e')
        fig2.tight_layout()
        st.pyplot(fig2)

    with col2:
        st.subheader("12-Month Rolling Sharpe Ratio")
        if not rolling_sharpe.empty:
            fig3, ax3 = plt.subplots(figsize=(6, 3), facecolor='#1e1e1e')
            ax3.plot(rolling_sharpe.index, rolling_sharpe, color='orange')
            ax3.axhline(0, linestyle='--', color='white', alpha=0.4)
            ax3.set_title("Rolling Sharpe Ratio", color='white')
            ax3.tick_params(colors='white')
            ax3.grid(False)
            ax3.set_facecolor('#1e1e1e')
            fig3.patch.set_facecolor('#1e1e1e')
            fig3.tight_layout()
            st.pyplot(fig3)
        else:
            st.warning("Not enough data for rolling Sharpe ratio.")

    st.subheader("Performance Summary")
    st.dataframe(summary_df.round(4), use_container_width=True)

    return summary_df
