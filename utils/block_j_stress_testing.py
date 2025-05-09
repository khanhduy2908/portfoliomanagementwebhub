import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

    weights = np.array(list(best_portfolio['Weights'].values()))
    tickers = list(best_portfolio['Weights'].keys())
    beta_dict = latest_data.set_index('Ticker')['Beta'].to_dict()

    df_price = data_stocks[data_stocks['Ticker'].isin(tickers)].copy()
    df_price['time'] = pd.to_datetime(df_price['time'], errors='coerce')
    df_pivot = df_price.pivot(index='time', columns='Ticker', values='Close').sort_index()
    df_pivot = df_pivot.dropna(axis=1, how='any')

    valid_tickers = [t for t in tickers if t in df_pivot.columns]
    if len(valid_tickers) < len(tickers):
        st.warning(f"Some tickers missing in price data: {set(tickers) - set(valid_tickers)}")

    tickers = valid_tickers
    weights = np.array([best_portfolio['Weights'][t] for t in tickers])
    monthly_returns = df_pivot[tickers].pct_change().dropna()

    mu_vec = monthly_returns.mean().values
    cov_matrix = monthly_returns.cov().values

    def generate_auto_shocks(tickers, beta_dict, base_rate_shock, inflation_shock):
        scenarios = {
            "Interest Rate Shock": {},
            "Tech Crash": {},
            "Inflation Shock": {}
        }
        for t in tickers:
            beta = beta_dict.get(t, 1.0)
            scenarios["Interest Rate Shock"][t] = base_rate_shock * beta
            scenarios["Inflation Shock"][t] = inflation_shock * beta
            if beta >= 1.2:
                scenarios["Tech Crash"][t] = -0.25
            elif beta >= 1.0:
                scenarios["Tech Crash"][t] = -0.15
        return scenarios

    scenario_map = generate_auto_shocks(tickers, beta_dict, shock_scale_interest, shock_scale_inflation)

    hypo_results = []
    for name, shock_map in scenario_map.items():
        shock_vector = np.array([shock_map.get(t, 0) for t in tickers])
        port_ret = np.dot(weights, shock_vector)
        hypo_results.append({'Scenario': name, 'Portfolio Return (%)': port_ret * 100})

    historical_shock = -0.25
    stress_replay = np.array([beta_dict.get(t, 1.0) * historical_shock for t in tickers])
    portfolio_drop_hist = np.dot(weights, stress_replay)

    sim_stress = np.random.multivariate_normal(mu_vec, cov_matrix, size=n_simulations)
    sim_stress += t_dist.rvs(t_dist_df, size=(n_simulations, len(tickers))) * 0.02
    returns_sim = sim_stress @ weights
    stress_var = -np.percentile(returns_sim, 100 - confidence_level * 100)
    stress_cvar = -returns_sim[returns_sim <= -stress_var].mean()

    sensitivity_results = []
    for i, t in enumerate(tickers):
        v = np.zeros(len(tickers))
        v[i] = -0.20
        impact = np.dot(weights, v)
        sensitivity_results.append({'Ticker': t, 'Portfolio Impact (%)': impact * 100})

    # === Combined layout for F1–F3 ===
    fig1, ax1 = plt.subplots(figsize=(5, 4), facecolor='#1e1e1e')
    sns.barplot(data=df_hypo, x='Scenario', y='Portfolio Return (%)', palette='Reds', edgecolor='black', ax=ax1)
    ax1.axhline(0, linestyle='--', color='white')
    ax1.set_title("Scenario Impact", color='white')
    ax1.set_xlabel("Scenario", color='white')
    ax1.set_ylabel("Portfolio Return (%)", color='white')
    ax1.tick_params(colors='white')
    ax1.set_facecolor('#1e1e1e')
    for label in ax1.get_xticklabels(): label.set_color('white')
    for label in ax1.get_yticklabels(): label.set_color('white')

    fig2, ax2 = plt.subplots(figsize=(5, 4), facecolor='#1e1e1e')
    sns.barplot(data=df_sens, x='Ticker', y='Portfolio Impact (%)', palette='Blues', edgecolor='black', ax=ax2)
    ax2.axhline(0, linestyle='--', color='white')
    ax2.set_title("Asset Sensitivity", color='white')
    ax2.set_xlabel("Ticker", color='white')
    ax2.set_ylabel("Portfolio Impact (%)", color='white')
    ax2.tick_params(colors='white')
    ax2.set_facecolor('#1e1e1e')
    for label in ax2.get_xticklabels(): label.set_color('white')
    for label in ax2.get_yticklabels(): label.set_color('white')

    fig3, ax3 = plt.subplots(figsize=(5, 4), facecolor='#1e1e1e')
    sns.histplot(returns_sim * 100, bins=50, kde=True, color='purple', ax=ax3)
    ax3.axvline(-stress_var * 100, color='red', linestyle='--', label=f"VaR {int(confidence_level*100)}%: {-stress_var*100:.2f}%")
    ax3.axvline(-stress_cvar * 100, color='orange', linestyle='--', label=f"CVaR {int(confidence_level*100)}%: {-stress_cvar*100:.2f}%")
    ax3.set_title("Monte Carlo Return Dist.", color='white')
    ax3.set_xlabel("Portfolio Return (%)", color='white')
    ax3.set_ylabel("Frequency", color='white')
    ax3.tick_params(colors='white')
    ax3.set_facecolor('#1e1e1e')
    ax3.legend(facecolor='black', labelcolor='white')
    
    st.markdown("### Stress Testing Overview")
    col1, col2, col3 = st.columns(3)
    with col1: st.pyplot(fig1, clear_figure=True)
    with col2: st.pyplot(fig2, clear_figure=True)
    with col3: st.pyplot(fig3, clear_figure=True)

    # === Summary Table ===
    summary = pd.DataFrame({
        'Type': ['Historical Shock', f'Monte Carlo VaR ({int(confidence_level*100)}%)', f'Monte Carlo CVaR ({int(confidence_level*100)}%)'],
        'Portfolio Drop (%)': [portfolio_drop_hist * 100, stress_var * 100, stress_cvar * 100],
        'Generated At': datetime.now().strftime('%Y-%m-%d %H:%M')
    })

    st.dataframe(summary.round(2), use_container_width=True)

    return summary
