import streamlit as st
import pandas as pd
import plotly.express as px

def run(returns_df, tickers, rf_monthly, start_date, end_date):
    st.subheader("Asset-Level Risk & Performance")

    # Lọc dữ liệu theo khoảng thời gian
    df = returns_df[tickers].copy()
    df = df.loc[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]

    if df.empty:
        st.error("No return data available for the selected date range.")
        return

    # Tính toán các chỉ số tài chính
    monthly_returns = df / 100
    cum_returns = (1 + monthly_returns).cumprod()
    ann_returns = monthly_returns.mean() * 12
    ann_volatility = monthly_returns.std() * (12 ** 0.5)
    sharpe_ratios = (ann_returns - rf_monthly) / ann_volatility

    # Biểu đồ Cumulative Return
    cum_returns.index.name = 'Date'
    cum_returns_reset = cum_returns.reset_index().melt(id_vars='Date', var_name='Ticker', value_name='Cumulative Return')
    fig1 = px.line(
        cum_returns_reset,
        x='Date',
        y='Cumulative Return',
        color='Ticker',
        title='Cumulative Returns',
        labels={'Date': 'Date', 'Cumulative Return': 'Growth Index'}
    )
    fig1.update_layout(
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        font_color='white',
        legend=dict(bgcolor='#1e1e1e')
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Biểu đồ Risk-Return Bubble Chart
    df_metrics = pd.DataFrame({
        'Ticker': tickers,
        'Annual Return (%)': ann_returns * 100,
        'Annual Volatility (%)': ann_volatility * 100,
        'Sharpe Ratio': sharpe_ratios
    })
    fig2 = px.scatter(
        df_metrics,
        x='Annual Volatility (%)',
        y='Annual Return (%)',
        size='Sharpe Ratio',
        color='Sharpe Ratio',
        hover_name='Ticker',
        color_continuous_scale=px.colors.sequential.Viridis,
        title='Risk vs Return',
        labels={'Annual Volatility (%)': 'Volatility (%)', 'Annual Return (%)': 'Return (%)'}
    )
    fig2.update_layout(
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        font_color='white',
        coloraxis_colorbar=dict(title="Sharpe Ratio"),
        legend=dict(bgcolor='#1e1e1e')
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Biểu đồ Rolling Sharpe Ratio (12 tháng) với trục x nghiêng -40 độ
    rolling_sharpe = (monthly_returns - rf_monthly).rolling(12).mean() / monthly_returns.rolling(12).std()
    rolling_sharpe = rolling_sharpe.dropna()

    if not rolling_sharpe.empty:
        rolling_sharpe.index.name = 'Date'
        rolling_df = rolling_sharpe.reset_index().melt(id_vars='Date', var_name='Ticker', value_name='Rolling Sharpe Ratio')
        fig3 = px.line(
            rolling_df,
            x='Date',
            y='Rolling Sharpe Ratio',
            color='Ticker',
            title='12-Month Rolling Sharpe Ratio',
            labels={'Date': 'Date', 'Rolling Sharpe Ratio': 'Rolling Sharpe Ratio'}
        )
        fig3.update_layout(
            plot_bgcolor='#1e1e1e',
            paper_bgcolor='#1e1e1e',
            font_color='white',
            legend=dict(bgcolor='#1e1e1e'),
            xaxis=dict(tickangle=-40)
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("Not enough data for rolling Sharpe ratio.")
