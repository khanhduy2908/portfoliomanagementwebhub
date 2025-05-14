import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def run(data_stocks, benchmark_data, benchmark_symbol, portfolio_weights, tickers,
        start_date, end_date, allocations_time=None):

    st.subheader("Portfolio Dynamics & Risk Analysis")

    # Chuẩn hóa dữ liệu thời gian
    data_stocks['time'] = pd.to_datetime(data_stocks['time'])
    benchmark_data['time'] = pd.to_datetime(benchmark_data['time'])

    # Lọc dữ liệu theo khoảng thời gian
    mask_stocks = (data_stocks['time'] >= start_date) & (data_stocks['time'] <= end_date)
    mask_bench = (benchmark_data['time'] >= start_date) & (benchmark_data['time'] <= end_date)
    df_stocks = data_stocks.loc[mask_stocks]
    df_bench = benchmark_data.loc[mask_bench]

    # Pivot dữ liệu giá
    price_stocks = df_stocks.pivot(index='time', columns='Ticker', values='Close').sort_index()
    price_bench = df_bench.pivot(index='time', columns='Ticker', values='Close').sort_index()

    # Tính returns
    returns_stocks = price_stocks.pct_change().dropna()
    returns_bench = price_bench[benchmark_symbol].pct_change().dropna()

    # Chỉ lấy ngày chung
    common_dates = returns_stocks.index.intersection(returns_bench.index)
    returns_stocks = returns_stocks.loc[common_dates]
    returns_bench = returns_bench.loc[common_dates]

    # Rolling window size
    window = 30

    # Tính beta động
    port_returns = returns_stocks.dot(portfolio_weights)
    rolling_cov = port_returns.rolling(window).cov(returns_bench)
    rolling_var = returns_bench.rolling(window).var()
    rolling_beta = rolling_cov / rolling_var

    # Historical drawdown danh mục
    cumulative_returns = (1 + port_returns).cumprod()
    rolling_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / rolling_max - 1

    # Rolling Volatility & Rolling Sharpe
    rolling_vol = port_returns.rolling(window).std() * np.sqrt(252)  # Annualized volatility
    risk_free_rate = 0  # Nếu có rf thực tế, thay đổi ở đây
    rolling_mean = port_returns.rolling(window).mean() * 252
    rolling_sharpe = (rolling_mean - risk_free_rate) / rolling_vol

    # Phân bổ tài sản theo thời gian (nếu có)
    # allocations_time: DataFrame có cột ['time', 'Ticker', 'Allocation']
    has_allocation = allocations_time is not None and not allocations_time.empty

    # Tạo figure với 3 hàng 2 cột
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Rolling Beta (30d)", "Historical Drawdown",
            "Rolling Volatility (Annualized)", "Rolling Sharpe Ratio (Annualized)",
            "Return Distribution Histogram", "Asset Allocation Over Time"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.15
    )

    # Row 1, Col 1: Rolling Beta
    fig.add_trace(go.Scatter(x=rolling_beta.index, y=rolling_beta, mode='lines', name='Rolling Beta'),
                  row=1, col=1)

    # Row 1, Col 2: Historical Drawdown
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown, mode='lines', name='Drawdown',
                             line=dict(color='red')),
                  row=1, col=2)

    # Row 2, Col 1: Rolling Volatility
    fig.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol, mode='lines', name='Rolling Volatility'),
                  row=2, col=1)

    # Row 2, Col 2: Rolling Sharpe Ratio
    fig.add_trace(go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe, mode='lines', name='Rolling Sharpe'),
                  row=2, col=2)

    # Row 3, Col 1: Return Distribution Histogram
    fig.add_trace(go.Histogram(x=port_returns, nbinsx=50, name='Return Distribution', marker_color='purple'),
                  row=3, col=1)

    # Row 3, Col 2: Asset Allocation Over Time hoặc thông báo không có dữ liệu
    if has_allocation:
        alloc_pivot = allocations_time.pivot(index='time', columns='Ticker', values='Allocation').fillna(0)
        for ticker in alloc_pivot.columns:
            fig.add_trace(go.Bar(
                x=alloc_pivot.index,
                y=alloc_pivot[ticker],
                name=ticker,
                showlegend=True
            ), row=3, col=2)
        fig.update_layout(barmode='stack')
    else:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            name="No allocation data", showlegend=False
        ), row=3, col=2)

    # Cài đặt chung layout
    fig.update_layout(
        height=900,
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        font_color='white',
        legend=dict(bgcolor='#1e1e1e', font=dict(color='white')),
        margin=dict(t=50, b=30, l=40, r=40),
        title_text="Advanced Portfolio Risk & Allocation Dynamics",
        title_x=0.5
    )

    # Trục x, y màu trắng
    for axis in fig.layout:
        if isinstance(fig.layout[axis], go.layout.XAxis) or isinstance(fig.layout[axis], go.layout.YAxis):
            fig.layout[axis].update(color='white')

    st.plotly_chart(fig, use_container_width=True)
