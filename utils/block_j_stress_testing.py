import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t
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
    df_price['time'] = pd.to_datetime(df_price['time'])
    df_pivot = df_price.pivot(index='time', columns='Ticker', values='Close').sort_index()
    monthly_returns = df_pivot.pct_change().dropna()

    mu_vec = np.array([monthly_returns[t].mean() for t in tickers])
    cov_matrix = monthly_returns[tickers].cov().values

    def generate_auto_shocks(tickers, beta_dict, base_rate_shock, inflation_shock):
        shock_dict = {
            "Interest Rate Shock": {},
            "Tech Crash": {},
            "Inflation Shock": {}
        }
        for t in tickers:
            beta = beta_dict.get(t, 1.0)
            shock_dict["Interest Rate Shock"][t] = base_rate_shock * beta
            shock_dict["Inflation Shock"][t] = inflation_shock * beta
            if beta >= 1.2:
                shock_dict["Tech Crash"][t] = -0.25
            elif beta >= 1.0:
                shock_dict["Tech Crash"][t] = -0.15
        return shock_dict

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
    sim_stress += t.rvs(t_dist_df, size=(n_simulations, len(tickers))) * 0.02
    returns_sim = sim_stress @ weights
    stress_var = -np.percentile(returns_sim, 100 - confidence_level * 100)
    stress_cvar = -returns_sim[returns_sim <= -stress_var].mean()

    sensitivity_results = []
    for i, t in enumerate(tickers):
        v = np.zeros(len(tickers))
        v[i] = -0.20
        impact = np.dot(weights, v)
        sensitivity_results.append({'Ticker': t, 'Portfolio Impact (%)': impact * 100})

    st.subheader("Auto Stress Scenario Impact")
    df_hypo = pd.DataFrame(hypo_results)
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df_hypo, x='Scenario', y='Portfolio Return (%)', palette='Reds', edgecolor='black', ax=ax1)
    ax1.axhline(0, linestyle='--', color='black')
    ax1.set_title("Auto Stress Scenario Impact on Portfolio", fontsize=14)
    st.pyplot(fig1)

    st.subheader("Sensitivity to 20% Drop in Each Asset")
    df_sens = pd.DataFrame(sensitivity_results)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df_sens, x='Ticker', y='Portfolio Impact (%)', palette='Blues', edgecolor='black', ax=ax2)
    ax2.axhline(0, linestyle='--', color='black')
    ax2.set_title("Sensitivity: 20% Drop per Asset", fontsize=14)
    st.pyplot(fig2)

    st.subheader("Monte Carlo Simulation Stress Distribution")
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    sns.histplot(returns_sim * 100, bins=50, kde=True, color='purple', ax=ax3)
    ax3.axvline(-stress_var * 100, color='red', linestyle='--', label=f"VaR {int(confidence_level*100)}%: {-stress_var*100:.2f}%")
    ax3.axvline(-stress_cvar * 100, color='orange', linestyle='--', label=f"CVaR {int(confidence_level*100)}%: {-stress_cvar*100:.2f}%")
    ax3.set_title("Monte Carlo Stress Distribution", fontsize=14)
    ax3.set_xlabel("Portfolio Return (%)")
    ax3.legend()
    st.pyplot(fig3)

    summary = pd.DataFrame({
        'Type': ['Historical Shock', 'Monte Carlo VaR', 'Monte Carlo CVaR'],
        'Portfolio Drop (%)': [portfolio_drop_hist * 100, stress_var * 100, stress_cvar * 100],
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M')
    })

    st.subheader("Stress Testing Summary")
    st.dataframe(summary.round(2))
