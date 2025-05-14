import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

    st.markdown("### Stress Testing Overview")

    # --- Plot Scenario Impact ---
    fig1 = px.bar(
        df_hypo,
        x='Scenario',
        y='Portfolio Return (%)',
        color='Portfolio Return (%)',
        color_continuous_scale='Reds',
        title='Scenario Impact',
        labels={'Portfolio Return (%)': 'Portfolio Return (%)'}
    )
    fig1.update_layout(
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        font_color='white',
        coloraxis_colorbar=dict(title='Return (%)')
    )
    st.plotly_chart(fig1, use_container_width=True)

    # --- Plot Asset Sensitivity ---
    fig2 = px.bar(
        df_sens,
        x='Ticker',
        y='Portfolio Impact (%)',
        color='Portfolio Impact (%)',
        color_continuous_scale='Blues',
        title='Asset Sensitivity',
        labels={'Portfolio Impact (%)': 'Portfolio Impact (%)'}
    )
    fig2.update_layout(
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        font_color='white',
        coloraxis_colorbar=dict(title='Impact (%)')
    )
    st.plotly_chart(fig2, use_container_width=True)

    # --- Plot Monte Carlo Return Distribution ---
    df_hist = pd.DataFrame({'Returns (%)': returns_sim * 100})
    fig3 = px.histogram(
        df_hist,
        x='Returns (%)',
        nbins=50,
        marginal='rug',
        title='Monte Carlo Return Distribution',
        color_discrete_sequence=['purple']
    )
    fig3.add_vline(
        x=-stress_var * 100,
        line_dash='dash',
        line_color='red',
        annotation_text=f"VaR {int(confidence_level*100)}%: {-stress_var*100:.2f}%",
        annotation_position='top right'
    )
    fig3.add_vline(
        x=-stress_cvar * 100,
        line_dash='dash',
        line_color='orange',
        annotation_text=f"CVaR {int(confidence_level*100)}%: {-stress_cvar*100:.2f}%",
        annotation_position='top right'
    )
    fig3.update_layout(
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        font_color='white',
    )
    st.plotly_chart(fig3, use_container_width=True)

    # --- Summary Table ---
    summary = pd.DataFrame({
        'Type': ['Historical Shock', f'Monte Carlo VaR ({int(confidence_level*100)}%)', f'Monte Carlo CVaR ({int(confidence_level*100)}%)'],
        'Portfolio Drop (%)': [portfolio_drop_hist * 100, stress_var * 100, stress_cvar * 100],
        'Generated At': datetime.now().strftime('%Y-%m-%d %H:%M')
    })
    st.dataframe(summary.round(2), use_container_width=True)

    return summary
