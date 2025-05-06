import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import streamlit as st

plt.style.use('dark_background')

def run(data_stocks, data_benchmark, benchmark_symbol,
        weights, tickers_portfolio,
        start_date, end_date, rf):

    st.subheader("Portfolio vs Benchmark Comparison")

    # STEP 1: Extract and align price data
    df_price = data_stocks[data_stocks['Ticker'].isin(tickers_portfolio)].copy()
    df_benchmark = data_benchmark[data_benchmark['Ticker'] == benchmark_symbol].copy()

    df_price['time'] = pd.to_datetime(df_price['time'], errors='coerce')
    df_benchmark['time'] = pd.to_datetime(df_benchmark['time'], errors='coerce')

    df_prices = df_price.pivot(index='time', columns='Ticker', values='Close').sort_index()
    df_benchmark = df_benchmark.pivot(index='time', columns='Ticker', values='Close').sort_index()
    df_benchmark = df_benchmark[[benchmark_symbol]]

    common_dates = df_prices.index.intersection(df_benchmark.index)
    if common_dates.empty:
        st.warning("No overlapping dates between portfolio and benchmark.")
        return

    df_prices = df_prices.loc[common_dates]
    df_benchmark = df_benchmark.loc[common_dates]

    # STEP 2: Compute returns
    portfolio_returns = df_prices.pct_change().dropna() @ weights
    benchmark_returns = df_benchmark[benchmark_symbol].pct_change().dropna()

    cum_portfolio = (1 + portfolio_returns).cumprod()
    cum_benchmark = (1 + benchmark_returns).cumprod()

    # STEP 3: Annualized metrics
    mean_p, mean_b = portfolio_returns.mean() * 12, benchmark_returns.mean() * 12
    vol_p, vol_b = portfolio_returns.std() * np.sqrt(12), benchmark_returns.std() * np.sqrt(12)
    sharpe_p = (mean_p - rf * 12) / vol_p if vol_p > 0 else np.nan
    sharpe_b = (mean_b - rf * 12) / vol_b if vol_b > 0 else np.nan

    # Layout: 2 columns
    col1, col2 = st.columns(2)

    # --- Chart 1: Cumulative Return ---
    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(cum_portfolio.index, cum_portfolio * 100, label='Portfolio', color='dodgerblue', linewidth=2)
        ax1.plot(cum_benchmark.index, cum_benchmark * 100, label=benchmark_symbol, color='crimson', linewidth=2)
        ax1.set_title("Cumulative Return")
        ax1.set_ylabel("Cumulative Return (%)")
        ax1.set_xlabel("Time")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)

    # --- Chart 2: Risk vs Return Bubble ---
    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4))

        labels = ['Portfolio', benchmark_symbol]
        vols = [vol_p, vol_b]
        rets = [mean_p, mean_b]
        sharpes = [sharpe_p, sharpe_b]

        norm = Normalize(vmin=min(sharpes), vmax=max(sharpes))
        cmap = cm.coolwarm
        colors = cmap(norm(sharpes))

        for i in range(2):
            x = vols[i] * 100
            y = rets[i] * 100
            color = colors[i]
            ax2.scatter(x, y, s=1000, c=[color], edgecolors='white')

            luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            text_color = 'black' if luminance > 0.6 else 'white'

            ax2.annotate(
                f"{labels[i]}\nSharpe: {sharpes[i]:.2f}",
                (x, y),
                ha='center',
                va='center',
                fontsize=9,
                color=text_color,
                weight='bold'
            )

        ax2.set_title("Risk vs Return (Annualized)")
        ax2.set_xlabel("Volatility (%)")
        ax2.set_ylabel("Return (%)")
        ax2.grid(True, alpha=0.3)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax2)
        cbar.set_label('Sharpe Ratio')

        st.pyplot(fig2)
