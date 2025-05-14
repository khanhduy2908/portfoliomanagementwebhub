import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import t as t_dist
from datetime import datetime

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

    # === Plotly Figures ===
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Scenario Impact", "Asset Sensitivity", "Monte Carlo Return Distribution"),
        horizontal_spacing=0.1
    )

    # Scenario Impact Bar Chart
    fig.add_trace(go.Bar(
        x=df_hypo['Scenario'],
        y=df_hypo['Portfolio Return (%)'],
        marker_color='crimson',
        name='Scenario Impact',
        text=df_hypo['Portfolio Return (%)'].apply(lambda x: f"{x:.2f}%"),
        textposition='auto'
    ), row=1, col=1)

    # Asset Sensitivity Bar Chart
    fig.add_trace(go.Bar(
        x=df_sens['Ticker'],
        y=df_sens['Portfolio Impact (%)'],
        marker_color='royalblue',
        name='Asset Sensitivity',
        text=df_sens['Portfolio Impact (%)'].apply(lambda x: f"{x:.2f}%"),
        textposition='auto'
    ), row=1, col=2)

    # Monte Carlo Return Histogram with VaR and CVaR lines
    fig.add_trace(go.Histogram(
        x=returns_sim * 100,
        nbinsx=50,
        name='Return Distribution',
        marker_color='purple',
        opacity=0.8
    ), row=1, col=3)

    # Add VaR and CVaR vertical lines
    fig.add_vline(
        x=-stress_var * 100,
        line=dict(color='red', width=3, dash='dash'),
        annotation_text=f"VaR {int(confidence_level*100)}%: {-stress_var*100:.2f}%",
        annotation_position="top left"
    )
    fig.add_vline(
        x=-stress_cvar * 100,
        line=dict(color='orange', width=3, dash='dash'),
        annotation_text=f"CVaR {int(confidence_level*100)}%: {-stress_cvar*100:.2f}%",
        annotation_position="top right"
    )

    # Layout and style
    fig.update_layout(
        height=450,
        plot_bgcolor='#121212',
        paper_bgcolor='#121212',
        font=dict(color='white'),
        showlegend=False,
        margin=dict(t=60, b=40, l=30, r=30)
    )

    fig.update_xaxes(title_text="Portfolio Return (%)", row=1, col=3, color='white')
    fig.update_yaxes(title_text="Frequency", row=1, col=3, color='white')

    fig.update_xaxes(color='white')
    fig.update_yaxes(color='white')

    st.markdown("### Stress Testing Overview")
    st.plotly_chart(fig, use_container_width=True)

    # === Summary Table ===
    summary = pd.DataFrame({
        'Type': ['Historical Shock', f'Monte Carlo VaR ({int(confidence_level*100)}%)', f'Monte Carlo CVaR ({int(confidence_level*100)}%)'],
        'Portfolio Drop (%)': [portfolio_drop_hist * 100, stress_var * 100, stress_cvar * 100],
        'Generated At': datetime.now().strftime('%Y-%m-%d %H:%M')
    })
    st.dataframe(summary.round(2), use_container_width=True)

    return summary
