import numpy as np
import pandas as pd
import streamlit as st

def run(best_portfolio, returns_pivot_stocks, returns_benchmark,
        rf, A, total_capital,
        data_stocks, data_benchmark, benchmark_symbol,
        weights, tickers_portfolio, start_date, end_date):

    # --- Chuẩn hoá index, chọn khung thời gian ---
    returns_pivot_stocks.index = pd.to_datetime(returns_pivot_stocks.index)
    returns_benchmark.index = pd.to_datetime(returns_benchmark.index)
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    common_index = date_range.intersection(returns_pivot_stocks.index)

    if common_index.empty:
        st.error("❌ No return data available for the selected date range.")
        return None

    returns_pivot_stocks = returns_pivot_stocks.loc[common_index]
    returns_benchmark = returns_benchmark.loc[common_index]

    # --- Tính toán lợi nhuận danh mục và benchmark ---
    portfolio_returns = returns_pivot_stocks[tickers_portfolio] @ weights
    benchmark_returns = returns_benchmark['Benchmark_Return']

    # --- Tính tích luỹ lợi nhuận ---
    cumulative_returns = (1 + portfolio_returns / 100).cumprod()
    benchmark_cumulative = (1 + benchmark_returns / 100).cumprod()
    cumulative_returns /= cumulative_returns.iloc[0]
    benchmark_cumulative /= benchmark_cumulative.iloc[0]

    # --- Các chỉ số hiệu suất ---
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

    # --- Rolling Sharpe Ratio 12 tháng ---
    rolling_sharpe = (portfolio_returns - rf * 100).rolling(12).mean() / portfolio_returns.rolling(12).std()
    rolling_sharpe = rolling_sharpe.dropna()

    # --- Tóm tắt hiệu suất so với benchmark ---
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

    # Chỉ trả về summary, không vẽ biểu đồ matplotlib ở đây
    return summary_df, rolling_sharpe
