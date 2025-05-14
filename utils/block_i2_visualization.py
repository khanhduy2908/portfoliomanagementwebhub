import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def run(data_stocks, benchmark_data, benchmark_symbol, portfolio_weights, tickers,
        start_date, end_date):

    st.subheader("Portfolio Dynamics & Risk Analysis")

    # Chuẩn hóa thời gian và dữ liệu
    data_stocks['time'] = pd.to_datetime(data_stocks['time'])
    benchmark_data['time'] = pd.to_datetime(benchmark_data['time'])

    # Lọc dữ liệu theo thời gian
    mask_stocks = (data_stocks['time'] >= start_date) & (data_stocks['time'] <= end_date)
    mask_bench = (benchmark_data['time'] >= start_date) & (benchmark_data['time'] <= end_date)

    df_stocks = data_stocks.loc[mask_stocks]
    df_bench = benchmark_data.loc[mask_bench]

    # Pivot data price
    price_stocks = df_stocks.pivot(index='time', columns='Ticker', values='Close').sort_index()
    price_bench = df_bench.pivot(index='time', columns='Ticker', values='Close').sort_index()

    # Tính returns
    returns_stocks = price_stocks.pct_change().dropna()
    returns_bench = price_bench[benchmark_symbol].pct_change().dropna()

    # Cắt dữ liệu đồng thời
    common_dates = returns_stocks.index.intersection(returns_bench.index)
    returns_stocks = returns_stocks.loc[common_dates]
    returns_bench = returns_bench.loc[common_dates]

    # --- Rolling Beta (30 ngày) ---
    window = 30
    rolling_beta = pd.Series(dtype=float, index=common_dates)

    # Tính beta danh mục theo công thức beta = cov(Rp, Rb) / var(Rb)
    port_returns = returns_stocks.dot(portfolio_weights)
    rolling_cov = port_returns.rolling(window).cov(returns_bench)
    rolling_var = returns_bench.rolling(window).var()
    rolling_beta = rolling_cov / rolling_var

    # --- Historical Drawdown danh mục ---
    cumulative_returns = (1 + port_returns).cumprod()
    rolling_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / rolling_max - 1

    # --- Rolling Volatility và Rolling Sharpe ---
    rolling_vol = port_returns.rolling(window).std() * np.sqrt(252)  # Annualized volatility
    risk_free_rate = 0  # Thay đổi nếu có rf
    rolling_mean = port_returns.rolling(window).mean() * 252
    rolling_sharpe = (rolling_mean - risk_free_rate) / rolling_vol

    # --- Asset Allocation Over Time ---
    # Giả sử có data phân bổ từng tài sản theo thời gian (ví dụ DataFrame allocations_time với columns: 'time', 'Ticker', 'Allocation')
    # Nếu không có, bỏ qua hoặc tạo ví dụ giả lập
    allocations_time = None  # Bạn cần truyền dữ liệu này nếu có

    # --- Return Distribution Histogram ---
    # Biểu đồ histogram lợi nhuận danh mục
    hist_data = port_returns

    # === Tạo Figure với nhiều subplot ===
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Rolling Beta (30d)", "Historical Drawdown",
            "Rolling Volatility (Annualized)", "Rolling Sharpe Ratio (Annualized)",
            "Return Distribution Histogram", "Asset Allocation Over Time"
        ),
        vertical_spacing=0.1,
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
    fig.add_trace(go.Histogram(x=hist_data, nbinsx=50, name='Return Distribution', marker_color='purple'),
                  row=3, col=1)

    # Row 3, Col 2: Asset Allocation Over Time (nếu có)
    if allocations_time is not None and not allocations_time.empty:
        # Cần dữ liệu allocations_time: columns = ['time', 'Ticker', 'Allocation']
        # Chuyển sang wide format cho stacked bar
        alloc_pivot = allocations_time.pivot(index='time', columns='Ticker', values='Allocation').fillna(0)
        for ticker in alloc_pivot.columns:
            fig.add_trace(go.Bar(
                x=alloc_pivot.index,
                y=alloc_pivot[ticker],
                name=ticker,
                marker=dict(),
                showlegend=True
            ), row=3, col=2)
        fig.update_layout(barmode='stack')
    else:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            name="No allocation data", showlegend=False
        ), row=3, col=2)

    # Layout chung
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

    # Trục x và y màu trắng cho tất cả subplot
    for axis in fig.layout:
        if isinstance(fig.layout[axis], go.layout.XAxis) or isinstance(fig.layout[axis], go.layout.YAxis):
            fig.layout[axis].update(color='white')

    st.plotly_chart(fig, use_container_width=True)
