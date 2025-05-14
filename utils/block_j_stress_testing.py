import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy.stats import t as t_dist
from datetime import datetime

def run(best_portfolio, latest_data, data_stocks, returns_pivot_stocks, rf):
    # --- Sidebar inputs for interactivity ---
    confidence_level = st.sidebar.slider("Confidence Level (%)", 90, 99, 95) / 100
    n_simulations = st.sidebar.number_input("Number of Monte Carlo Simulations", min_value=1000, max_value=50000, value=10000, step=1000)

    # Shock scales and parameters
    shock_scale_interest = -0.15
    shock_scale_inflation = -0.10
    t_dist_df = 4
    np.random.seed(42)

    # Extract weights and tickers
    weights_all = best_portfolio['Weights']
    tickers_all = list(weights_all.keys())
    beta_dict = latest_data.set_index('Ticker')['Beta'].to_dict()

    # Prepare price data
    df_price = data_stocks[data_stocks['Ticker'].isin(tickers_all)].copy()
    df_price['time'] = pd.to_datetime(df_price['time'], errors='coerce')
    df_pivot = df_price.pivot(index='time', columns='Ticker', values='Close').sort_index().dropna(axis=1, how='any')

    # Filter tickers available in price data
    tickers = [t for t in tickers_all if t in df_pivot.columns]
    weights = np.array([weights_all[t] for t in tickers])
    beta_dict = {t: beta_dict.get(t, 1.0) for t in tickers}

    # Calculate returns mean and covariance
    monthly_returns = df_pivot[tickers].pct_change().dropna()
    mu_vec = monthly_returns.mean().values
    cov_matrix = monthly_returns.cov().values

    # Generate stress scenarios based on beta and predefined shocks
    def generate_auto_shocks(tickers, beta_dict, base_shock, infl_shock):
        scenarios = {
            "Interest Rate Shock": {},
            "Tech Crash": {},
            "Inflation Shock": {}
        }
        for t in tickers:
            beta = beta_dict.get(t, 1.0)
            scenarios["Interest Rate Shock"][t] = base_shock * beta
            scenarios["Inflation Shock"][t] = infl_shock * beta
            scenarios["Tech Crash"][t] = -0.25 if beta >= 1.2 else -0.15 if beta >= 1.0 else 0
        return scenarios

    scenario_map = generate_auto_shocks(tickers, beta_dict, shock_scale_interest, shock_scale_inflation)
    hypo_results = [
        {'Scenario': name, 'Portfolio Return (%)': np.dot(weights, np.array([shock.get(t, 0) for t in tickers])) * 100}
        for name, shock in scenario_map.items()
    ]
    df_hypo = pd.DataFrame(hypo_results)

    # Historical shock impact
    hist_shock = np.array([beta_dict.get(t, 1.0) * -0.25 for t in tickers])
    portfolio_drop_hist = np.dot(weights, hist_shock)

    # Monte Carlo simulation with t-distribution shocks
    sim_stress = np.random.multivariate_normal(mu_vec, cov_matrix, size=n_simulations)
    sim_stress += t_dist.rvs(t_dist_df, size=(n_simulations, len(tickers))) * 0.02
    returns_sim = sim_stress @ weights
    stress_var = -np.percentile(returns_sim, 100 - confidence_level * 100)
    stress_cvar = -returns_sim[returns_sim <= -stress_var].mean()

    # Asset sensitivity to a hypothetical -20% shock
    sensitivity_results = [
        {'Ticker': t, 'Portfolio Impact (%)': -0.20 * weights[i] * 100}
        for i, t in enumerate(tickers)
    ]
    df_sens = pd.DataFrame(sensitivity_results)

    # === Plot 1: Scenario Impact ===
    st.markdown("### Stress Testing Overview")
    fig1 = px.bar(df_hypo, x='Scenario', y='Portfolio Return (%)', color='Portfolio Return (%)',
                  color_continuous_scale='Reds', labels={'Portfolio Return (%)': 'Return (%)'},
                  title='Scenario Impact')
    st.plotly_chart(fig1, use_container_width=True)

    # === Plot 2: Asset Sensitivity ===
    fig2 = px.bar(df_sens, x='Ticker', y='Portfolio Impact (%)', color='Portfolio Impact (%)',
                  color_continuous_scale='Blues', labels={'Portfolio Impact (%)': 'Impact (%)'},
                  title='Asset Sensitivity')
    st.plotly_chart(fig2, use_container_width=True)

    # === Plot 3: Monte Carlo Return Distribution ===
    fig3 = px.histogram(returns_sim * 100, nbins=50, title='Monte Carlo Return Distribution', color_discrete_sequence=['purple'])
    fig3.add_vline(x=-stress_var * 100, line_dash='dash', line_color='red', annotation_text=f"VaR {int(confidence_level*100)}%: {-stress_var*100:.2f}%", annotation_position="top left")
    fig3.add_vline(x=-stress_cvar * 100, line_dash='dash', line_color='orange', annotation_text=f"CVaR {int(confidence_level*100)}%: {-stress_cvar*100:.2f}%", annotation_position="top left")
    st.plotly_chart(fig3, use_container_width=True)

    # === Summary Table ===
    summary = pd.DataFrame({
        'Type': ['Historical Shock', f'Monte Carlo VaR ({int(confidence_level*100)}%)', f'Monte Carlo CVaR ({int(confidence_level*100)}%)'],
        'Portfolio Drop (%)': [portfolio_drop_hist * 100, stress_var * 100, stress_cvar * 100],
        'Generated At': datetime.now().strftime('%Y-%m-%d %H:%M')
    })

    st.dataframe(summary.round(2), use_container_width=True)

    return summary
