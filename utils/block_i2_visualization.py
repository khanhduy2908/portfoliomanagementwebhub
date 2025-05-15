import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def run(data_stocks, benchmark_data, benchmark_symbol, portfolio_weights, tickers,
        start_date, end_date, rf):

    st.subheader("Portfolio Dynamics & Risk Analysis")

    # Chuẩn hóa thời gian
    data_stocks['time'] = pd.to_datetime(data_stocks['time'])
    benchmark_data['time'] = pd.to_datetime(benchmark_data['time'])

    # Lọc dữ liệu
    mask_stocks = (data_stocks['time'] >= pd.to_datetime(start_date)) & (data_stocks['time'] <= pd.to_datetime(end_date))
    mask_bench = (benchmark_data['time'] >= pd.to_datetime(start_date)) & (benchmark_data['time'] <= pd.to_datetime(end_date))
    df_stocks = data_stocks.loc[mask_stocks]
    df_bench = benchmark_data.loc[mask_bench]

    price_stocks = df_stocks.pivot(index='time', columns='Ticker', values='Close').sort_index()
    price_bench = df_bench.pivot(index='time', columns='Ticker', values='Close').sort_index()

    returns_stocks = price_stocks.pct_change().dropna()
    returns_bench = price_bench[benchmark_symbol].pct_change().dropna()

    # Đồng bộ ngày
    common_dates = returns_stocks.index.intersection(returns_bench.index)
    returns_stocks = returns_stocks.loc[common_dates]
    returns_bench = returns_bench.loc[common_dates]

    # Khớp ticker và trọng số
    valid_tickers = [t for t in tickers if t in returns_stocks.columns]
    returns_stocks = returns_stocks[valid_tickers]
    weights = np.array([portfolio_weights[tickers.index(t)] for t in valid_tickers])
    port_returns = returns_stocks.dot(weights)

    # Tính toán rolling
    window = 30
    rolling_beta = port_returns.rolling(window).cov(returns_bench) / returns_bench.rolling(window).var()
    cumulative_returns = (1 + port_returns).cumprod()
    drawdown = cumulative_returns / cumulative_returns.cummax() - 1
    rolling_vol = port_returns.rolling(window).std() * np.sqrt(252)
    rolling_mean = port_returns.rolling(window).mean() * 252
    rolling_sharpe = (rolling_mean - rf) / rolling_vol
    hist_data = port_returns

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "Rolling Beta (30d)", "Historical Drawdown",
            "Rolling Volatility (Annualized)", "Rolling Sharpe Ratio (Annualized)",
            "Return Distribution Histogram", "Asset Allocation Over Time"
        ],
        vertical_spacing=0.1, horizontal_spacing=0.12
    )

    # Row 1
    fig.add_trace(go.Scatter(x=rolling_beta.index, y=rolling_beta, mode='lines', name='Rolling Beta'), row=1, col=1)
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown, mode='lines', name='Drawdown', line=dict(color='red')), row=1, col=2)

    # Row 2
    fig.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol, mode='lines', name='Volatility'), row=2, col=1)
    fig.add_trace(go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe, mode='lines', name='Sharpe Ratio'), row=2, col=2)

    # Row 3
    fig.add_trace(go.Histogram(x=hist_data, nbinsx=50, marker_color='purple', name='Return Dist.'), row=3, col=1)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name="No allocation data", showlegend=False), row=3, col=2)

    fig.update_layout(
        height=900,
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        font_color='white',
        title="Advanced Portfolio Risk & Allocation Dynamics",
        title_x=0.5,
        margin=dict(t=50, b=40, l=50, r=40)
    )

    # Áp dụng màu trục trắng
    for axis in fig.layout:
        if isinstance(fig.layout[axis], go.layout.XAxis) or isinstance(fig.layout[axis], go.layout.YAxis):
            fig.layout[axis].update(color='white')

    st.plotly_chart(fig, use_container_width=True)
