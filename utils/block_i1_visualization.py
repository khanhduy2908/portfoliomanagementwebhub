import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

def run(returns_df, tickers, rf_monthly, start_date, end_date):
    st.subheader("Asset-Level Risk & Performance")

    returns_df.index = pd.to_datetime(returns_df.index)
    df = returns_df[tickers].copy()
    df = df.loc[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]

    if df.empty:
        st.error("No return data available for the selected date range.")
        return

    # Tính toán các chỉ số
    monthly_returns = df / 100
    cum_returns = (1 + monthly_returns).cumprod()
    ann_returns = monthly_returns.mean() * 12
    ann_volatility = monthly_returns.std() * np.sqrt(12)
    sharpe_ratios = (ann_returns - rf_monthly) / ann_volatility

    # Drawdown
    cumulative_max = cum_returns.cummax()
    drawdown = cum_returns / cumulative_max - 1

    # Rolling Sharpe Ratio (12 tháng)
    rolling_sharpe = (monthly_returns - rf_monthly).rolling(12).mean() / monthly_returns.rolling(12).std()
    rolling_sharpe = rolling_sharpe.dropna()

    # Rolling Volatility (12 tháng)
    rolling_volatility = monthly_returns.rolling(12).std() * np.sqrt(12)
    rolling_volatility = rolling_volatility.dropna()

    # Contribution to Risk (Marginal contribution by variance)
    cov_matrix = monthly_returns.cov()
    portfolio_weights = np.array([1/len(tickers)]*len(tickers))  # Cân bằng giả định, bạn có thể truyền tham số weights nếu có
    port_variance = portfolio_weights.T @ cov_matrix.values @ portfolio_weights
    # Đóng góp từng tài sản theo công thức MCR = w_i * (Cov_i * w) / Portfolio Variance
    mcr = portfolio_weights * (cov_matrix.values @ portfolio_weights) / port_variance
    contribution_risk = mcr / mcr.sum()

    df_contrib_risk = pd.DataFrame({
        'Ticker': tickers,
        'Risk Contribution (%)': contribution_risk * 100
    }).sort_values(by='Risk Contribution (%)', ascending=False)

    # Correlation heatmap
    corr_matrix = monthly_returns.corr()

    # === Bố cục 3 hàng 2 cột ===
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)
    row3_col1, row3_col2 = st.columns(2)

    # Cumulative Return
    with row1_col1:
        cum_returns_reset = cum_returns.reset_index().melt(id_vars='index', var_name='Ticker', value_name='Cumulative Return')
        fig_cum = px.line(
            cum_returns_reset,
            x='index', y='Cumulative Return', color='Ticker',
            title='Cumulative Returns',
            labels={'index': 'Date', 'Cumulative Return': 'Growth Index'}
        )
        fig_cum.update_layout(plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e', font_color='white', legend=dict(bgcolor='#1e1e1e'))
        st.plotly_chart(fig_cum, use_container_width=True)

    # Drawdown
    with row1_col2:
        drawdown_reset = drawdown.reset_index().melt(id_vars='index', var_name='Ticker', value_name='Drawdown')
        fig_dd = px.area(
            drawdown_reset,
            x='index', y='Drawdown', color='Ticker',
            title='Drawdown (%)',
            labels={'index': 'Date', 'Drawdown': 'Drawdown'}
        )
        fig_dd.update_layout(plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e', font_color='white', legend=dict(bgcolor='#1e1e1e'))
        st.plotly_chart(fig_dd, use_container_width=True)

    # Rolling Sharpe Ratio
    with row2_col1:
        rolling_sharpe_reset = rolling_sharpe.reset_index().melt(id_vars='index', var_name='Ticker', value_name='Rolling Sharpe Ratio')
        fig_rs = px.line(
            rolling_sharpe_reset,
            x='index', y='Rolling Sharpe Ratio', color='Ticker',
            title='12-Month Rolling Sharpe Ratio',
            labels={'index': 'Date', 'Rolling Sharpe Ratio': 'Rolling Sharpe Ratio'}
        )
        fig_rs.update_layout(plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e', font_color='white', legend=dict(bgcolor='#1e1e1e'),
                             xaxis=dict(tickangle=-40))
        st.plotly_chart(fig_rs, use_container_width=True)

    # Rolling Volatility
    with row2_col2:
        rolling_vol_reset = rolling_volatility.reset_index().melt(id_vars='index', var_name='Ticker', value_name='Rolling Volatility')
        fig_rv = px.line(
            rolling_vol_reset,
            x='index', y='Rolling Volatility', color='Ticker',
            title='12-Month Rolling Volatility',
            labels={'index': 'Date', 'Rolling Volatility': 'Rolling Volatility'}
        )
        fig_rv.update_layout(plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e', font_color='white', legend=dict(bgcolor='#1e1e1e'),
                             xaxis=dict(tickangle=-40))
        st.plotly_chart(fig_rv, use_container_width=True)

    # Risk Contribution Bar Chart
    with row3_col1:
        fig_risk = px.bar(
            df_contrib_risk,
            x='Ticker', y='Risk Contribution (%)',
            title='Contribution to Portfolio Risk (%)',
            labels={'Risk Contribution (%)': 'Risk Contribution (%)'}
        )
        fig_risk.update_layout(plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e', font_color='white')
        st.plotly_chart(fig_risk, use_container_width=True)

    # Correlation Heatmap
    with row3_col2:
        heatmap = ff.create_annotated_heatmap(
            z=corr_matrix.values.round(2),
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.index.tolist(),
            colorscale='RdBu',
            reversescale=True,
            zmin=-1, zmax=1,
            showscale=True,
            font_colors=['white']
        )
        heatmap.update_layout(plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e', font_color='white',
                              title_text='Correlation Heatmap')
        st.plotly_chart(heatmap, use_container_width=True)
