import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import t as t_dist
from datetime import datetime
import streamlit as st

def plot_monte_carlo_return_distribution(returns_sim, stress_var, stress_cvar, confidence_level):
    df_hist = pd.DataFrame({'Returns (%)': returns_sim * 100})

    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=df_hist['Returns (%)'],
        nbinsx=50,
        marker_color='purple',
        opacity=0.75,
        name='Simulated Returns'
    ))

    # VaR line
    fig.add_shape(
        type="line",
        x0=-stress_var * 100, y0=0,
        x1=-stress_var * 100, y1=df_hist['Returns (%)'].value_counts().max(),
        line=dict(color="red", width=3, dash="dash"),
    )
    fig.add_annotation(
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
    fig.add_shape(
        type="line",
        x0=-stress_cvar * 100, y0=0,
        x1=-stress_cvar * 100, y1=df_hist['Returns (%)'].value_counts().max(),
        line=dict(color="orange", width=3, dash="dash"),
    )
    fig.add_annotation(
        x=-stress_cvar * 100,
        y=df_hist['Returns (%)'].value_counts().max()*0.85,
        text=f"CVaR {int(confidence_level*100)}%: {-stress_cvar*100:.2f}%",
        showarrow=True,
        arrowhead=3,
        ax=40,
        ay=-40,
        font=dict(color="orange", size=12)
    )

    fig.update_layout(
        title="Monte Carlo Return Distribution",
        xaxis_title="Portfolio Return (%)",
        yaxis_title="Frequency",
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        font_color='white',
        bargap=0.05,
        hovermode='x unified'
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)

    return fig

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

    weights = np.array(list(best_portfolio['Weights'].values()))
    tickers = list(best_portfolio['Weights'].keys())

    beta_dict = latest_data.set_index('Ticker')['Beta'].to_dict()

    df_price = data_stocks[data_stocks['Ticker'].isin(tickers)].copy()
    df_price['time'] = pd.to_datetime(df_price['time'], errors='coerce')
    df_pivot = df_price.pivot(index='time', columns='Ticker', values='Close').sort_index().dropna(axis=1, how='any')

    tickers = [t for t in tickers if t in df_pivot.columns]
    weights = np.array([best_portfolio['Weights'][t] for t in tickers])

    monthly_returns = df_pivot[tickers].pct_change().dropna()
    mu_vec = monthly_returns.mean().values
    cov_matrix = monthly_returns.cov().values

    # Generate Shock Scenarios
    def generate_auto_shocks(tickers, beta_dict, base_shock, infl_shock):
        scenarios = {"Interest Rate Shock": {}, "Tech Crash": {}, "Inflation Shock": {}}
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

    # Historical Shock
    hist_shock = np.array([beta_dict.get(t, 1.0) * -0.25 for t in tickers])
    portfolio_drop_hist = np.dot(weights, hist_shock)

    # Monte Carlo Simulation
    sim_stress = np.random.multivariate_normal(mu_vec, cov_matrix, size=n_simulations)
    sim_stress += t_dist.rvs(t_dist_df, size=(n_simulations, len(tickers))) * 0.02
    returns_sim = sim_stress @ weights
    stress_var = -np.percentile(returns_sim, 100 - confidence_level * 100)
    stress_cvar = -returns_sim[returns_sim <= -stress_var].mean()

    # Sensitivity Test
    sensitivity_results = [
        {'Ticker': t, 'Portfolio Impact (%)': -0.20 * weights[i] * 100}
        for i, t in enumerate(tickers)
    ]
    df_sens = pd.DataFrame(sensitivity_results)

    st.markdown("### Stress Testing Overview")
    col1, col2, col3 = st.columns(3)

    # Plot Scenario Impact
    fig1 = px.bar(df_hypo, x='Scenario', y='Portfolio Return (%)',
                  color='Portfolio Return (%)', color_continuous_scale='Reds',
                  title="Scenario Impact")
    fig1.update_layout(
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        font_color='white',
        xaxis_title="Scenario",
        yaxis_title="Portfolio Return (%)",
        coloraxis_showscale=False,
    )
    col1.plotly_chart(fig1, use_container_width=True)

    # Plot Asset Sensitivity
    fig2 = px.bar(df_sens, x='Ticker', y='Portfolio Impact (%)',
                  color='Portfolio Impact (%)', color_continuous_scale='Blues',
                  title="Asset Sensitivity")
    fig2.update_layout(
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        font_color='white',
        xaxis_title="Ticker",
        yaxis_title="Portfolio Impact (%)",
        coloraxis_showscale=False,
    )
    col2.plotly_chart(fig2, use_container_width=True)

    # Plot Monte Carlo Distribution
    fig3 = plot_monte_carlo_return_distribution(returns_sim, stress_var, stress_cvar, confidence_level)
    col3.plotly_chart(fig3, use_container_width=True)

    # Summary Table
    summary = pd.DataFrame({
        'Type': ['Historical Shock', f'Monte Carlo VaR ({int(confidence_level*100)}%)', f'Monte Carlo CVaR ({int(confidence_level*100)}%)'],
        'Portfolio Drop (%)': [portfolio_drop_hist * 100, stress_var * 100, stress_cvar * 100],
        'Generated At': datetime.now().strftime('%Y-%m-%d %H:%M')
    })
    st.dataframe(summary.round(2), use_container_width=True)

    return summary
