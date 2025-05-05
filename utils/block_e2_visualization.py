# utils/block_e2_visualization.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import streamlit as st
from matplotlib.colors import Normalize

def run(data_stocks, data_benchmark, benchmark_symbol,
        weights, tickers_portfolio,
        start_date, end_date, rf):

    st.markdown("### 📊 Portfolio vs Benchmark Visualization")

    # --- STEP 1: Chuẩn bị dữ liệu giá ---
    df_price = data_stocks[data_stocks['Ticker'].isin(tickers_portfolio)].copy()
    df_benchmark = data_benchmark[data_benchmark['Ticker'] == benchmark_symbol].copy()

    df_price['time'] = pd.to_datetime(df_price['time'], errors='coerce')
    df_benchmark['time'] = pd.to_datetime(df_benchmark['time'], errors='coerce')

    df_prices = df_price.pivot(index='time', columns='Ticker', values='Close').sort_index()
    df_benchmark = df_benchmark.pivot(index='time', columns='Ticker', values='Close').sort_index()
    df_benchmark = df_benchmark[[benchmark_symbol]]

    common_dates = df_prices.index.intersection(df_benchmark.index)
    if common_dates.empty:
        st.warning("⚠️ Không có ngày giao dịch chung giữa danh mục và benchmark.")
        return

    df_prices = df_prices.loc[common_dates]
    df_benchmark = df_benchmark.loc[common_dates]

    # --- STEP 2: Tính toán lợi suất và tích lũy ---
    portfolio_returns = df_prices.pct_change().dropna() @ weights
    benchmark_returns = df_benchmark[benchmark_symbol].pct_change().dropna()

    cum_portfolio = (1 + portfolio_returns).cumprod()
    cum_benchmark = (1 + benchmark_returns).cumprod()

    # --- STEP 3: Risk & Return ---
    mean_p, mean_b = portfolio_returns.mean() * 12, benchmark_returns.mean() * 12
    vol_p, vol_b = portfolio_returns.std() * np.sqrt(12), benchmark_returns.std() * np.sqrt(12)
    sharpe_p = (mean_p - rf * 12) / vol_p if vol_p > 0 else np.nan
    sharpe_b = (mean_b - rf * 12) / vol_b if vol_b > 0 else np.nan

    # --- STEP 4: Vẽ biểu đồ hiệu suất tích lũy ---
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(cum_portfolio.index, cum_portfolio * 100, label='Portfolio', color='dodgerblue', linewidth=2)
    ax1.plot(cum_benchmark.index, cum_benchmark * 100, label=benchmark_symbol, color='crimson', linewidth=2)
    ax1.set_title("Cumulative Return Comparison", fontsize=14)
    ax1.set_ylabel("Cumulative Return (%)")
    ax1.set_xlabel("Time")
    ax1.legend()
    ax1.grid(False)
    st.pyplot(fig1)

    # --- STEP 5: Biểu đồ Risk-Return Bubble ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    labels = ['Portfolio', benchmark_symbol]
    vols = [vol_p, vol_b]
    rets = [mean_p, mean_b]
    sharpes = [sharpe_p, sharpe_b]

    norm = Normalize(vmin=min(sharpes), vmax=max(sharpes))
    colors = cm.coolwarm(norm(sharpes))

    for i in range(2):
        x = vols[i] * 100
        y = rets[i] * 100
        color = colors[i]
        ax2.scatter(x, y, s=1500, c=[color], edgecolors='black', label=labels[i])

        # Hiển thị tên + Sharpe ratio ngay trên bubble
        ax2.annotate(
            f"{labels[i]}\nSharpe: {sharpes[i]:.2f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=10,
            color='black',
            weight='bold'
        )

    ax2.set_title("Risk vs Return (Annualized)", fontsize=14)
    ax2.set_xlabel("Volatility (%)")
    ax2.set_ylabel("Return (%)")
    ax2.grid(False)

    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2)
    cbar.set_label('Sharpe Ratio')

    st.pyplot(fig2)
