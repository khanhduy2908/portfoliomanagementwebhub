# utils/block_j_stress_testing.py

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import t as t_dist
from datetime import datetime
import streamlit as st

def run(best_portfolio, latest_data, data_stocks, returns_pivot_stocks, rf):
    st.markdown("## Stress Testing Overview", unsafe_allow_html=True)

    # === Params ===
    confidence_level = 0.95
    shock_scale_interest = -0.15
    shock_scale_inflation = -0.10
    t_dist_df = 4
    n_simulations = 10000
    np.random.seed(42)

    weights = np.array(list(best_portfolio['Weights'].values()))
    tickers = list(best_portfolio['Weights'].keys())

    beta_dict = latest_data.set_index('Ticker')['Beta'].to_dict()

    # Prepare data
    df_price = data_stocks[data_stocks['Ticker'].isin(tickers)].copy()
    df_price['time'] = pd.to_datetime(df_price['time'], errors='coerce')
    df_pivot = df_price.pivot(index='time', columns='Ticker', values='Close').sort_index().dropna(axis=1, how='any')

    tickers = [t for t in tickers if t in df_pivot.columns]
    weights = np.array([best_portfolio['Weights'][t] for t in tickers])

    monthly_returns = df_pivot[tickers].pct_change().dropna()
    mu_vec = monthly_returns.mean().values
    cov_matrix = monthly_returns.cov().values

    # === Scenarios ===
    def generate_auto_shocks(tickers, beta_dict, base_shock, infl_shock):
        return {
            "Interest Rate Shock": {t: base_shock * beta_dict.get(t, 1.0) for t in tickers},
            "Inflation Shock": {t: infl_shock * beta_dict.get(t, 1.0) for t in tickers},
            "Tech Crash": {t: -0.25 if beta_dict.get(t, 1.0) >= 1.2 else -0.15 if beta_dict.get(t, 1.0) >= 1.0 else 0 for t in tickers}
        }

    scenario_map = generate_auto_shocks(tickers, beta_dict, shock_scale_interest, shock_scale_inflation)
    df_hypo = pd.DataFrame([
        {'Scenario': k, 'Portfolio Return (%)': np.dot(weights, [v.get(t, 0) for t in tickers]) * 100}
        for k, v in scenario_map.items()
    ])

    # === Historical Shock ===
    hist_shock = np.array([beta_dict.get(t, 1.0) * -0.25 for t in tickers])
    portfolio_drop_hist = np.dot(weights, hist_shock)

    # === Monte Carlo Simulation ===
    sim_stress = np.random.multivariate_normal(mu_vec, cov_matrix, size=n_simulations)
    sim_stress += t_dist.rvs(t_dist_df, size=(n_simulations, len(tickers))) * 0.02
    returns_sim = sim_stress @ weights
    stress_var = -np.percentile(returns_sim, 100 - confidence_level * 100)
    stress_cvar = -returns_sim[returns_sim <= -stress_var].mean()

    # === Sensitivity Test ===
    df_sens = pd.DataFrame({
        'Ticker': tickers,
        'Portfolio Impact (%)': -0.2 * weights * 100
    })

    # === Visualization ===
    col1, col2, col3 = st.columns(3)

    with col1:
        fig1 = px.bar(
            df_hypo, x='Scenario', y='Portfolio Return (%)', color='Portfolio Return (%)',
            color_continuous_scale='Reds', range_y=[min(df_hypo['Portfolio Return (%)'].min(), -30), 0],
            title="Scenario Impact"
        )
        fig1.update_layout(
            plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e', font_color='white',
            coloraxis_colorbar=dict(title='Return %'), yaxis=dict(tickformat=".1f")
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.bar(
            df_sens, x='Ticker', y='Portfolio Impact (%)', color='Portfolio Impact (%)',
            color_continuous_scale='Blues_r', range_y=[df_sens['Portfolio Impact (%)'].min() * 1.2, 0],
            title="Asset Sensitivity"
        )
        fig2.update_layout(
            plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e', font_color='white',
            coloraxis_colorbar=dict(title='Impact %'), yaxis=dict(tickformat=".1f")
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        fig3 = px.histogram(
            x=returns_sim * 100, nbins=60, title="Monte Carlo Return Distribution",
            labels={'x': 'Portfolio Return (%)', 'count': 'Frequency'}, color_discrete_sequence=['purple']
        )
        fig3.add_vline(x=-stress_var * 100, line_dash='dash', line_color='red')
        fig3.add_vline(x=-stress_cvar * 100, line_dash='dash', line_color='orange')
        fig3.add_annotation(x=-stress_var * 100, y=0, text=f"VaR {int(confidence_level*100)}%:<br>{-stress_var*100:.2f}%",
                            showarrow=True, arrowhead=1, yanchor="bottom", font=dict(color="red"))
        fig3.add_annotation(x=-stress_cvar * 100, y=0, text=f"CVaR {int(confidence_level*100)}%:<br>{-stress_cvar*100:.2f}%",
                            showarrow=True, arrowhead=1, yanchor="top", font=dict(color="orange"))
        fig3.update_layout(
            plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e', font_color='white',
            margin=dict(t=40, b=40), xaxis=dict(tickformat=".1f")
        )
        st.plotly_chart(fig3, use_container_width=True)

    # === Summary Table ===
    st.markdown("### Stress Testing Summary")
    summary = pd.DataFrame({
        'Type': ['Historical Shock', f'Monte Carlo VaR ({int(confidence_level*100)}%)', f'Monte Carlo CVaR ({int(confidence_level*100)}%)'],
        'Portfolio Drop (%)': [portfolio_drop_hist * 100, stress_var * 100, stress_cvar * 100],
        'Generated At': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })
    st.dataframe(summary.round(2), use_container_width=True)

    return summary
