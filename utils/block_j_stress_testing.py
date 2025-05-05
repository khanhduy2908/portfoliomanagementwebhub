# utils/block_j_stress_testing.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime
from scipy.stats import t

def run(best_portfolio, latest_data, data_stocks, returns_pivot_stocks, rf):
    st.markdown("### 🧪 Multi-Layer Stress Testing")

    # --- Config ---
    confidence_level = 0.95
    shock_scale_interest = -0.15
    shock_scale_inflation = -0.10
    t_dist_df = 4
    n_simulations = 10000
    np.random.seed(42)

    # --- Extract info ---
    weights = np.array(list(best_portfolio['Weights'].values()))
    tickers = list(best_portfolio['Weights'].keys())
    beta_dict = latest_data.set_index('Ticker')['Beta'].to_dict()

    # --- Prepare monthly returns if needed ---
    df_price = data_stocks[data_stocks['Ticker'].isin(tickers)].copy()
    df_price['time'] = pd.to_datetime(df_price['time'])
    df_pivot = df_price.pivot(index='time', columns='Ticker', values='Close').sort_index()
    monthly_returns = df_pivot.pct_change().dropna()

    mu_vec = np.array([monthly_returns[t].mean() for t in tickers])
    cov_matrix = monthly_returns[tickers].cov().values

    # --- F.1: Hypothetical Shocks ---
    def generate_shocks(beta_dict):
        scenario_map = {
            "Interest Rate Shock": {},
            "Inflation Shock": {},
            "Tech Crash": {}
        }
        for t in tickers:
            beta = beta_dict.get(t, 1.0)
            scenario_map["Interest Rate Shock"][t] = shock_scale_interest * beta
            scenario_map["Inflation Shock"][t] = shock_scale_inflation * beta
            scenario_map["Tech Crash"][t] = -0.25 if beta >= 1.2 else -0.15 if beta >= 1.0 else -0.05
        return scenario_map

    scenario_map = generate_shocks(beta_dict)
    hypo_results = [
        {"Scenario": s, "Portfolio Return (%)": np.dot(weights, list(shock.values())) * 100}
        for s, shock in scenario_map.items()
    ]

    df_hypo = pd.DataFrame(hypo_results)
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    sns.barplot(data=df_hypo, x='Scenario', y='Portfolio Return (%)', palette='Reds', ax=ax1, edgecolor='black')
    ax1.axhline(0, linestyle='--', color='black')
    ax1.set_title("Auto Stress Scenario Impact")
    st.pyplot(fig1)

    # --- F.2: Historical Stress ---
    hist_shock = -0.25
    stress_replay = np.array([beta_dict.get(t, 1.0) * hist_shock for t in tickers])
    portfolio_drop_hist = np.dot(weights, stress_replay)

    # --- F.3: Monte Carlo Stress Simulation ---
    sim_stress = np.random.multivariate_normal(mu_vec, cov_matrix, size=n_simulations)
    sim_stress += t.rvs(t_dist_df, size=(n_simulations, len(tickers))) * 0.02
    returns_sim = sim_stress @ weights

    stress_var = -np.percentile(returns_sim, 100 - confidence_level * 100)
    stress_cvar = -returns_sim[returns_sim <= -stress_var].mean()

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.histplot(returns_sim * 100, bins=50, kde=True, color='purple', ax=ax2)
    ax2.axvline(-stress_var * 100, color='red', linestyle='--', label=f"VaR {int(confidence_level*100)}%: {-stress_var*100:.2f}%")
    ax2.axvline(-stress_cvar * 100, color='orange', linestyle='--', label=f"CVaR {int(confidence_level*100)}%: {-stress_cvar*100:.2f}%")
    ax2.set_title("Monte Carlo Stress Distribution")
    ax2.set_xlabel("Portfolio Return (%)")
    ax2.legend()
    st.pyplot(fig2)

    # --- F.4: Sensitivity Test ---
    sens_results = []
    for i, t in enumerate(tickers):
        v = np.zeros(len(tickers))
        v[i] = -0.2
        impact = np.dot(weights, v)
        sens_results.append({'Ticker': t, 'Portfolio Impact (%)': impact * 100})

    df_sens = pd.DataFrame(sens_results)
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df_sens, x='Ticker', y='Portfolio Impact (%)', palette='Blues', ax=ax3, edgecolor='black')
    ax3.axhline(0, linestyle='--', color='black')
    ax3.set_title("Sensitivity: 20% Drop per Asset")
    st.pyplot(fig3)

    # --- Summary Table ---
    summary_df = pd.DataFrame({
        'Type': ['Historical Shock', 'Monte Carlo VaR', 'Monte Carlo CVaR'],
        'Portfolio Drop (%)': [portfolio_drop_hist * 100, stress_var * 100, stress_cvar * 100],
        'Timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M')] * 3
    })
    st.markdown("#### 🔍 Stress Test Summary")
    st.dataframe(summary_df.round(2), use_container_width=True)
