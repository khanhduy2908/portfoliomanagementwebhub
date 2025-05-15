import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def run(returns_df, tickers, rf_monthly, start_date, end_date):
    st.subheader("Asset-Level Risk & Performance")

    # Chuẩn hóa index và lọc dữ liệu
    returns_df.index = pd.to_datetime(returns_df.index)
    df = returns_df.loc[(returns_df.index >= pd.to_datetime(start_date)) & (returns_df.index <= pd.to_datetime(end_date)), tickers].copy()

    if df.empty:
        st.error("No return data available for the selected date range.")
        return

    # Tính toán chỉ số
    monthly_returns = df / 100
    cum_returns = (1 + monthly_returns).cumprod()
    ann_returns = monthly_returns.mean() * 12
    ann_volatility = monthly_returns.std() * np.sqrt(12)
    sharpe_ratios = (ann_returns - rf_monthly) / ann_volatility

    # Drawdown
    cumulative_max = cum_returns.cummax()
    drawdown = cum_returns / cumulative_max - 1

    # Rolling
    rolling_sharpe = (monthly_returns - rf_monthly).rolling(12).mean() / monthly_returns.rolling(12).std()
    rolling_volatility = monthly_returns.rolling(12).std() * np.sqrt(12)

    # Contribution to Risk
    cov_matrix = monthly_returns.cov()
    portfolio_weights = np.array([1 / len(tickers)] * len(tickers))
    port_variance = portfolio_weights.T @ cov_matrix.values @ portfolio_weights
    mcr = portfolio_weights * (cov_matrix.values @ portfolio_weights) / port_variance
    contribution_risk = mcr / mcr.sum()
    df_contrib_risk = pd.DataFrame({
        'Ticker': tickers,
        'Risk Contribution (%)': contribution_risk * 100
    }).sort_values(by='Risk Contribution (%)', ascending=False)

    # Risk-Return DataFrame
    df_metrics = pd.DataFrame({
        'Ticker': tickers,
        'Annual Return (%)': ann_returns * 100,
        'Annual Volatility (%)': ann_volatility * 100,
        'Sharpe Ratio': sharpe_ratios
    }).dropna()

    # === Layout: 3 hàng, 2 cột ===
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)
    row3_col1, row3_col2 = st.columns(2)

    # --- 1. Cumulative Return
    with row1_col1:
        cum_returns.index.name = 'Date'
        cum_returns_reset = cum_returns.reset_index().melt(id_vars='Date', var_name='Ticker', value_name='Cumulative Return')
        fig = px.line(cum_returns_reset, x='Date', y='Cumulative Return', color='Ticker', title='Cumulative Returns')
        fig.update_layout(
            plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e', font_color='white',
            legend=dict(bgcolor='#1e1e1e'))
        st.plotly_chart(fig, use_container_width=True)

    # --- 2. Drawdown
    with row1_col2:
        drawdown.index.name = 'Date'
        drawdown_reset = drawdown.reset_index().melt(id_vars='Date', var_name='Ticker', value_name='Drawdown')
        fig = px.area(drawdown_reset, x='Date', y='Drawdown', color='Ticker', title='Drawdown (%)')
        fig.update_layout(
            plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e', font_color='white',
            legend=dict(bgcolor='#1e1e1e'))
        st.plotly_chart(fig, use_container_width=True)

    # --- 3. Rolling Sharpe Ratio
    with row2_col1:
        rolling_sharpe = rolling_sharpe.dropna()
        rolling_sharpe.index.name = 'Date'
        rolling_df = rolling_sharpe.reset_index().melt(id_vars='Date', var_name='Ticker', value_name='Rolling Sharpe Ratio')
        fig = px.line(rolling_df, x='Date', y='Rolling Sharpe Ratio', color='Ticker', title='12-Month Rolling Sharpe Ratio')
        fig.update_layout(
            plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e', font_color='white',
            xaxis=dict(tickangle=-40), legend=dict(bgcolor='#1e1e1e'))
        st.plotly_chart(fig, use_container_width=True)

    # --- 4. Rolling Volatility
    with row2_col2:
        rolling_volatility = rolling_volatility.dropna()
        rolling_volatility.index.name = 'Date'
        rolling_df = rolling_volatility.reset_index().melt(id_vars='Date', var_name='Ticker', value_name='Rolling Volatility')
        fig = px.line(rolling_df, x='Date', y='Rolling Volatility', color='Ticker', title='12-Month Rolling Volatility')
        fig.update_layout(
            plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e', font_color='white',
            xaxis=dict(tickangle=-40), legend=dict(bgcolor='#1e1e1e'))
        st.plotly_chart(fig, use_container_width=True)

    # --- 5. Contribution to Risk
    with row3_col1:
        fig = px.bar(df_contrib_risk, x='Ticker', y='Risk Contribution (%)',
                     color='Risk Contribution (%)', color_continuous_scale=px.colors.sequential.Plasma,
                     title='Contribution to Portfolio Risk (%)')
        fig.update_layout(
            plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e', font_color='white',
            coloraxis_colorbar=dict(title="Risk Contribution (%)"),
            legend=dict(bgcolor='#1e1e1e'))
        st.plotly_chart(fig, use_container_width=True)

    # --- 6. Risk vs Return Scatter (thay vì correlation)
    with row3_col2:
        fig = px.scatter(
            df_metrics,
            x='Annual Volatility (%)',
            y='Annual Return (%)',
            size='Sharpe Ratio',
            color='Sharpe Ratio',
            hover_name='Ticker',
            color_continuous_scale=px.colors.sequential.Viridis,
            title='Risk vs Return'
        )
        fig.update_layout(
            plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e', font_color='white',
            coloraxis_colorbar=dict(title="Sharpe Ratio"),
            legend=dict(bgcolor='#1e1e1e'))
        st.plotly_chart(fig, use_container_width=True)
