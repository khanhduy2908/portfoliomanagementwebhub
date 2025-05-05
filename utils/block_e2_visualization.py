import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import pandas as pd


def plot_e1_asset_performance_and_risk_bubble(returns_df, tickers, rf_monthly, start_date, end_date):
    df = returns_df[tickers].copy()
    df = df.loc[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]

    if df.empty:
        raise ValueError("❌ Không có dữ liệu sau khi lọc thời gian.")

    daily_returns = df / 100
    cum_returns = (1 + daily_returns).cumprod()

    annualized_returns = daily_returns.mean() * 12
    annualized_volatility = daily_returns.std() * np.sqrt(12)
    sharpe_ratios = (annualized_returns - rf_monthly) / annualized_volatility

    fig, ax = plt.subplots(1, 2, figsize=(16, 6), facecolor='white')

    for ticker in tickers:
        ax[0].plot(cum_returns.index, cum_returns[ticker], label=ticker, linewidth=2)
    ax[0].set_title("Historical Cumulative Performance", fontsize=13)
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Cumulative Return")
    ax[0].legend()
    ax[0].grid(False)

    colors = cm.coolwarm((sharpe_ratios - sharpe_ratios.min()) / (sharpe_ratios.max() - sharpe_ratios.min()))
    norm = Normalize(vmin=sharpe_ratios.min(), vmax=sharpe_ratios.max())
    cmap = cm.coolwarm

    for i, ticker in enumerate(tickers):
        color_val = colors[i]
        color_rgb = cmap(norm(sharpe_ratios[ticker]))
        luminance = 0.299 * color_rgb[0] + 0.587 * color_rgb[1] + 0.114 * color_rgb[2]
        text_color = 'black' if luminance > 0.6 else 'white'

        ax[1].scatter(
            annualized_volatility[ticker] * 100,
            annualized_returns[ticker] * 100,
            s=1500,
            c=[color_val],
            edgecolors='black',
            linewidths=1.5,
            alpha=0.95,
            zorder=3
        )
        ax[1].annotate(
            ticker,
            (annualized_volatility[ticker] * 100, annualized_returns[ticker] * 100),
            color=text_color,
            weight='bold',
            ha='center',
            va='center',
            fontsize=10,
            zorder=4
        )

    ax[1].set_title("Risk vs Return (Annualized)", fontsize=13)
    ax[1].set_xlabel("Volatility (%)")
    ax[1].set_ylabel("Return (%)")
    ax[1].grid(False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax[1])
    cbar.set_label('Sharpe Ratio')
    plt.show()
