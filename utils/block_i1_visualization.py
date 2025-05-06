import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import streamlit as st

def run(returns_df, tickers, rf_monthly, start_date, end_date):
    st.subheader("Asset-Level Risk & Performance")

    df = returns_df[tickers].copy()
    df = df.loc[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]

    if df.empty:
        st.error("No return data available for the selected date range.")
        return

    monthly_returns = df / 100
    cum_returns = (1 + monthly_returns).cumprod()
    ann_returns = monthly_returns.mean() * 12
    ann_volatility = monthly_returns.std() * np.sqrt(12)
    sharpe_ratios = (ann_returns - rf_monthly) / ann_volatility

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 4), facecolor='#1e1e1e')
        for ticker in tickers:
            ax1.plot(cum_returns.index, cum_returns[ticker], label=ticker, linewidth=1.8)
        ax1.set_title("Cumulative Returns", color='white')
        ax1.set_xlabel("Date", color='white')
        ax1.set_ylabel("Growth Index", color='white')
        ax1.legend(fontsize=8, facecolor='#1e1e1e', labelcolor='white')
        ax1.tick_params(colors='white')
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor('#1e1e1e')
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4), facecolor='#1e1e1e')

        cmap = cm.coolwarm
        norm = Normalize(vmin=sharpe_ratios.min(), vmax=sharpe_ratios.max())
        colors = cmap(norm(sharpe_ratios))

        for i, ticker in enumerate(tickers):
            color_rgb = colors[i]
            luminance = 0.299 * color_rgb[0] + 0.587 * color_rgb[1] + 0.114 * color_rgb[2]
            text_color = 'black' if luminance > 0.6 else 'white'

            ax2.scatter(
                ann_volatility[ticker] * 100,
                ann_returns[ticker] * 100,
                s=1000,
                c=[color_rgb],
                edgecolors='white',
                linewidths=1.5,
                alpha=0.9
            )
            ax2.annotate(
                ticker,
                (ann_volatility[ticker] * 100, ann_returns[ticker] * 100),
                color=text_color,
                ha='center',
                va='center',
                fontsize=9,
                weight='bold'
            )

        ax2.set_title("Risk vs Return", color='white')
        ax2.set_xlabel("Volatility (%)", color='white')
        ax2.set_ylabel("Return (%)", color='white')
        ax2.tick_params(colors='white')
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor('#1e1e1e')

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax2)
        cbar.set_label('Sharpe Ratio', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

        st.pyplot(fig2)
