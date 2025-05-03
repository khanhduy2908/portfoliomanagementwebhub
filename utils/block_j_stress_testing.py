import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import t

def run_block_j(best_portfolio, monthly_returns, beta_dict, rf, confidence_level=0.95,
                shock_scale_interest=-0.15, shock_scale_inflation=-0.10, t_dist_df=4, n_simulations=10000):
    np.random.seed(42)
    weights = np.array(list(best_portfolio['Weights'].values()))
    tickers = list(best_portfolio['Weights'].keys())

    mu_vec = np.array([monthly_returns[t].mean() for t in tickers])
    cov_matrix = monthly_returns[tickers].cov().values

    # F2: Scenario Shocks
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

    # F3: Historical Shock
    historical_shock = -0.25
    stress_replay = np.array([beta_dict.get(t, 1.0) * historical_shock for t in tickers])
    portfolio_drop_hist = np.dot(weights, stress_replay)

    # F4: Monte Carlo
    sim_stress = np.random.multivariate_normal(mu_vec, cov_matrix, size=n_simulations)
    sim_stress += t.rvs(t_dist_df, size=(n_simulations, len(tickers))) * 0.02
    returns_sim = sim_stress @ weights
    stress_var = -np.percentile(returns_sim, 100 - confidence_level * 100)
    stress_cvar = -returns_sim[returns_sim <= -stress_var].mean()

    # F5: Sensitivity
    sensitivity_results = []
    for i, t in enumerate(tickers):
        v = np.zeros(len(tickers))
        v[i] = -0.20
        impact = np.dot(weights, v)
        sensitivity_results.append({'Ticker': t, 'Portfolio Impact (%)': impact * 100})

    # F7: Summary Table
    summary = pd.DataFrame({
        'Type': ['Historical Shock', 'Monte Carlo VaR', 'Monte Carlo CVaR'],
        'Portfolio Drop (%)': [portfolio_drop_hist * 100, stress_var * 100, stress_cvar * 100],
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M')
    })

    return {
        "hypothetical_scenarios": pd.DataFrame(hypo_results),
        "sensitivity": pd.DataFrame(sensitivity_results),
        "monte_carlo_simulation": returns_sim * 100,
        "monte_carlo_var": stress_var * 100,
        "monte_carlo_cvar": stress_cvar * 100,
        "historical_shock": portfolio_drop_hist * 100,
        "summary_table": summary
    }