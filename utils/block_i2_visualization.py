import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.colors import Normalize
import matplotlib.cm as cm

def run(data_stocks, data_benchmark, benchmark_symbol,
        weights, tickers_portfolio,
        start_date, end_date, rf):

    st.subheader("Portfolio vs Benchmark Comparison")

    df_price = data_stocks[data_stocks['Ticker'].isin(tickers_portfolio)].copy()
    df_benchmark = data_benchmark[data_benchmark['Ticker'] == benchmark_symbol].copy()

    if df_price.empty or df_benchmark.empty:
        st.warning("Missing price data for portfolio or benchmark.")
        return

    df_price['time'] = pd.to_datetime(df_price['time'], errors='coerce')
    df_benchmark['time'] = pd.to_datetime(df_benchmark['time'], errors='coerce')

    df_prices = df_price.pivot(index='time', columns='Ticker', values='Close').sort_index()
    df_benchmark_pivot = df_benchmark.pivot(index='time', columns='Ticker', values='Close').sort_index()

    if benchmark_symbol not in df_benchmark_pivot.columns:
        st.warning("Benchmark symbol data not found.")
        return

    df_benchmark_pivot = df_benchmark_pivot[[benchmark_symbol]]

    common_dates = df_prices.index.intersection(df_benchmark_pivot.index)
    if common_dates.empty:
        st.warning("No overlapping dates between portfolio and benchmark.")
        return

    df_prices = df_prices.loc[common_dates]
    df_benchmark_pivot = df_benchmark_pivot.loc[common_dates]

    returns_matrix = df_prices.pct_change().dropna()
    portfolio_returns = returns_matrix @ weights
    portfolio_returns = pd.Series(portfolio_returns, index=returns_matrix.index)
    benchmark_returns = df_benchmark_pivot[benchmark_symbol].pct_change().dropna()

    if portfolio_returns.empty or benchmark_returns.empty:
        st.warning("Insufficient return data.")
        return

    cum_portfolio = (1 + portfolio_returns).cumprod()
    cum_benchmark = (1 + benchmark_returns).cumprod()

    mean_p = portfolio_returns.mean() * 12
    mean_b = benchmark_returns.mean() * 12
    vol_p = portfolio_returns.std() * np.sqrt(12)
    vol_b = benchmark_returns.std() * np.sqrt(12)
    sharpe_p = (mean_p - rf * 12) / vol_p if vol_p > 0 else np.nan
    sharpe_b = (mean_b - rf * 12) / vol_b if vol_b > 0 else np.nan

    # Rolling metrics
    rolling_cov = portfolio_returns.rolling(12).cov(benchmark_returns)
    rolling_var_benchmark = benchmark_returns.rolling(12).var()
    rolling_beta = (rolling_cov / rolling_var_benchmark).dropna()
    rolling_vol = portfolio_returns.rolling(12).std().dropna() * np.sqrt(12)
    rolling_sharpe = ((portfolio_returns.rolling(12).mean() * 12 - rf * 12) / rolling_vol).dropna()

    # Drawdown
    rolling_max = cum_portfolio.cummax()
    drawdown = cum_portfolio / rolling_max - 1

    # Color for risk vs return
    norm = Normalize(vmin=min(sharpe_p, sharpe_b), vmax=max(sharpe_p, sharpe_b))
    cmap = cm.coolwarm
    colors = [cmap(norm(val)) for val in [sharpe_p, sharpe_b]]
    colors_hex = [f'rgba({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)}, {c[3]:.2f})' for c in colors]

    # === Layout: 2 rows x 2 cols ===
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Cumulative Return (Normalized)",
            "Risk vs Return (Annualized)",
            "Drawdown",
            "Rolling Beta, Volatility, Sharpe"
        )
    )

    # Cumulative Return
    fig.add_trace(go.Scatter(x=cum_portfolio.index, y=cum_portfolio * 100,
                             mode='lines', name='Portfolio',
                             line=dict(color='dodgerblue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=cum_benchmark.index, y=cum_benchmark * 100,
                             mode='lines', name=benchmark_symbol,
                             line=dict(color='crimson')), row=1, col=1)

    # Risk vs Return
    fig.add_trace(go.Scatter(
        x=[vol_p * 100, vol_b * 100],
        y=[mean_p * 100, mean_b * 100],
        mode='markers+text',
        text=[f"Portfolio<br>Sharpe: {sharpe_p:.2f}", f"{benchmark_symbol}<br>Sharpe: {sharpe_b:.2f}"],
        textposition="top center",
        marker=dict(color=colors_hex, size=20, line=dict(width=2, color='white')),
        name='Risk vs Return'
    ), row=1, col=2)

    # Drawdown
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown * 100,
                             mode='lines', name='Drawdown',
                             fill='tozeroy', fillcolor='rgba(255,0,0,0.2)',
                             line=dict(color='red')), row=2, col=1)

    # Combined Rolling Metrics
    fig.add_trace(go.Scatter(x=rolling_beta.index, y=rolling_beta,
                             mode='lines', name='Rolling Beta',
                             line=dict(color='goldenrod')), row=2, col=2)
    fig.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol * 100,
                             mode='lines', name='Rolling Volatility',
                             line=dict(color='cyan')), row=2, col=2)
    fig.add_trace(go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe,
                             mode='lines', name='Rolling Sharpe',
                             line=dict(color='orange')), row=2, col=2)

    # Layout
    fig.update_layout(
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        font=dict(color='white'),
        height=750,
        margin=dict(t=60, b=20, l=30, r=30),
        showlegend=True
    )

    fig.update_xaxes(color='white')
    fig.update_yaxes(color='white')

    st.plotly_chart(fig, use_container_width=True)
