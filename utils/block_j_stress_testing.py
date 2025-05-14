# utils/block_j_stress_testing.py

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from scipy.stats import t as t_dist
from datetime import datetime
import streamlit as st

def run(best_portfolio, latest_data, data_stocks, returns_pivot_stocks, rf):
    confidence_level = 0.95
    shock_scale_interest = -0.15
    shock_scale_inflation = -0.10
    t_dist_df = 4
    n_simulations = 10000
    np.random.seed(42)

    # Reset index nếu chưa có cột index (tránh lỗi melt/pivot)
    if 'index' not in latest_data.columns:
        latest_data = latest_data.reset_index()
    if 'index' not in data_stocks.columns:
        data_stocks = data_stocks.reset_index()

    # Lấy trọng số và tickers danh mục
    weights = np.array(list(best_portfolio['Weights'].values()))
    tickers = list(best_portfolio['Weights'].keys())

    # Lấy beta theo ticker từ latest_data
    beta_dict = latest_data.set_index('Ticker')['Beta'].to_dict()

    # Chuẩn bị dữ liệu giá cho tickers
    df_price = data_stocks[data_stocks['Ticker'].isin(tickers)].copy()
    df_price['time'] = pd.to_datetime(df_price['time'], errors='coerce')
    df_pivot = df_price.pivot(index='time', columns='Ticker', values='Close').sort_index().dropna(axis=1, how='any')

    tickers = [t for t in tickers if t in df_pivot.columns]
    weights = np.array([best_portfolio['Weights'][t] for t in tickers])

    monthly_returns = df_pivot[tickers].pct_change().dropna()
    mu_vec = monthly_returns.mean().values
    cov_matrix = monthly_returns.cov().values

    # --- Generate Shock Scenarios ---
    def generate_auto_shocks(tickers, beta_dict, base_shock, infl_shock):
        scenarios = {"Interest Rate Shock": {}, "Tech Crash": {}, "Inflation Shock": {}}
        for t in tickers:
            beta = beta_dict.get(t, 1.0)
            scenarios["Interest Rate Shock"][t] = base_shock * beta
            scenarios["Inflation Shock"][t] = infl_shock * beta
            # Định nghĩa Tech Crash giảm sâu hơn cho beta lớn
            scenarios["Tech Crash"][t] = -0.25 if beta >= 1.2 else -0.15 if beta >= 1.0 else 0
        return scenarios

    scenario_map = generate_auto_shocks(tickers, beta_dict, shock_scale_interest, shock_scale_inflation)
    hypo_results = [
        {'Scenario': name, 'Portfolio Return (%)': np.dot(weights, np.array([shock.get(t, 0) for t in tickers])) * 100}
        for name, shock in scenario_map.items()
    ]
    df_hypo = pd.DataFrame(hypo_results)

    # --- Historical Shock ---
    hist_shock = np.array([beta_dict.get(t, 1.0) * -0.25 for t in tickers])
    portfolio_drop_hist = np.dot(weights, hist_shock)

    # --- Monte Carlo Simulation ---
    sim_stress = np.random.multivariate_normal(mu_vec, cov_matrix, size=n_simulations)
    sim_stress += t_dist.rvs(t_dist_df, size=(n_simulations, len(tickers))) * 0.02
    returns_sim = sim_stress @ weights
    stress_var = -np.percentile(returns_sim, 100 - confidence_level * 100)
    stress_cvar = -returns_sim[returns_sim <= -stress_var].mean()

    # --- Sensitivity Test ---
    sensitivity_results = [
        {'Ticker': t, 'Portfolio Impact (%)': -0.20 * weights[i] * 100}
        for i, t in enumerate(tickers)
    ]
    df_sens = pd.DataFrame(sensitivity_results)

    # === PLOTS with Plotly ===
    col1, col2, col3 = st.columns(3)

    with col1:
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
            x=df_hypo['Scenario'],
            y=df_hypo['Portfolio Return (%)'],
            marker_color=df_hypo['Portfolio Return (%)'].apply(lambda x: 'red' if x < 0 else 'green'),
            name='Scenario Impact'
        ))
        fig1.update_layout(
            title="Scenario Impact",
            plot_bgcolor='#1e1e1e',
            paper_bgcolor='#1e1e1e',
            font_color='white',
            yaxis_title="Portfolio Return (%)",
            xaxis_title="Scenario",
            xaxis_tickangle=-45,
            yaxis_gridcolor='#333333'
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=df_sens['Ticker'],
            y=df_sens['Portfolio Impact (%)'],
            marker_color=df_sens['Portfolio Impact (%)'].apply(lambda x: 'blue' if x < 0 else 'lightblue'),
            name='Asset Sensitivity'
        ))
        fig2.update_layout(
            title="Asset Sensitivity",
            plot_bgcolor='#1e1e1e',
            paper_bgcolor='#1e1e1e',
            font_color='white',
            yaxis_title="Portfolio Impact (%)",
            xaxis_title="Ticker",
            xaxis_tickangle=-45,
            yaxis_gridcolor='#333333'
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        df_hist = pd.DataFrame({'Returns (%)': returns_sim * 100})

        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(
            x=df_hist['Returns (%)'],
            nbinsx=50,
            marker_color='purple',
            opacity=0.75,
            name='Simulated Returns'
        ))

        # VaR line
        fig3.add_shape(
            type="line",
            x0=-stress_var * 100, y0=0,
            x1=-stress_var * 100, y1=df_hist['Returns (%)'].value_counts().max(),
            line=dict(color="red", width=3, dash="dash"),
        )
        fig3.add_annotation(
            x=-stress_var * 100,
            y=df_hist['Returns (%)'].value_counts().max(),
            text=f"VaR {int(confidence_level*100)}%: {-stress_var*100:.2f}%",
            showarrow=True,
            arrowhead=3,
            ax=40,
            ay=-40,
            font=dict(color="red", size=12)
        )

        # CVaR line
        fig3.add_shape(
            type="line",
            x0=-stress_cvar * 100, y0=0,
            x1=-stress_cvar * 100, y1=df_hist['Returns (%)'].value_counts().max() * 0.8,
            line=dict(color="orange", width=3, dash="dash"),
        )
        fig3.add_annotation(
            x=-stress_cvar * 100,
            y=df_hist['Returns (%)'].value_counts().max() * 0.8,
            text=f"CVaR {int(confidence_level*100)}%: {-stress_cvar*100:.2f}%",
            showarrow=True,
            arrowhead=3,
            ax=40,
            ay=-40,
            font=dict(color="orange", size=12)
        )

        fig3.update_layout(
            title="Monte Carlo Return Distribution",
            xaxis_title="Portfolio Return (%)",
            yaxis_title="Frequency",
            plot_bgcolor='#1e1e1e',
            paper_bgcolor='#1e1e1e',
            font_color='white',
            bargap=0.05,
            hovermode='x unified'
        )
        fig3.update_xaxes(showgrid=False, zeroline=False)
        fig3.update_yaxes(showgrid=False, zeroline=False)

        st.plotly_chart(fig3, use_container_width=True)

    # === Summary Table ===
    summary = pd.DataFrame({
        'Type': ['Historical Shock', f'Monte Carlo VaR ({int(confidence_level*100)}%)', f'Monte Carlo CVaR ({int(confidence_level*100)}%)'],
        'Portfolio Drop (%)': [portfolio_drop_hist * 100, stress_var * 100, stress_cvar * 100],
        'Generated At': datetime.now().strftime('%Y-%m-%d %H:%M')
    })

    st.dataframe(summary.round(2), use_container_width=True)

    return summary
