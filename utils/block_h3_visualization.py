import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

def run(
    best_portfolio, mu_p, sigma_p, rf, sigma_c, expected_rc, y_capped, y_opt,
    adj_returns_combinations=None, cov_matrix_dict=None, simulate_for_visual=True
):
    st.subheader("Efficient Frontier with Optimal Complete Portfolio")

    try:
        best_key = list(best_portfolio.keys())[0]
        tickers = list(best_key)
        result = best_portfolio[best_key]
        weights = np.array([result['Weights'][t] for t in tickers])
    except Exception as e:
        st.error(f"❌ Failed to extract best portfolio info: {e}")
        return

    if adj_returns_combinations and cov_matrix_dict:
        try:
            mu_dict = adj_returns_combinations.get(best_key)
            cov_df = cov_matrix_dict.get(best_key)
            if mu_dict is None or cov_df is None:
                raise ValueError("Missing mu or covariance data for best portfolio key.")
            mu_realistic = np.array([mu_dict[t] for t in tickers]) / 100
            cov = cov_df.loc[tickers, tickers].values
        except Exception as e:
            st.warning(f"⚠️ Using fallback mu/cov due to error: {e}")
            mu_realistic = np.full(len(tickers), result.get('Expected Return (%)', 0) / 100)
            cov = np.outer(weights, weights) * (result.get('Volatility (%)', 0) / 100) ** 2
    else:
        mu_realistic = np.full(len(tickers), result.get('Expected Return (%)', 0) / 100)
        cov = np.outer(weights, weights) * (result.get('Volatility (%)', 0) / 100) ** 2

    if simulate_for_visual:
        np.random.seed(42)
        n_simulations = 20000
        weights_sim = np.random.dirichlet(np.ones(len(tickers)), size=n_simulations)

        mu_sim = weights_sim @ mu_realistic
        sigma_sim = np.sqrt(np.einsum('ij,jk,ik->i', weights_sim, cov, weights_sim))
        sharpe_sim = (mu_sim - rf) / sigma_sim

        valid_mask = (sigma_sim > 1e-5) & (mu_sim > 1e-5) & np.isfinite(sharpe_sim)
        mu_sim, sigma_sim, sharpe_sim = mu_sim[valid_mask], sigma_sim[valid_mask], sharpe_sim[valid_mask]

        df_sim = pd.DataFrame({
            'Volatility (%)': sigma_sim * 100,
            'Expected Return (%)': mu_sim * 100,
            'Sharpe Ratio': sharpe_sim
        })
    else:
        df_sim = pd.DataFrame(columns=['Volatility (%)', 'Expected Return (%)', 'Sharpe Ratio'])

    fig = px.scatter(
        df_sim,
        x='Volatility (%)',
        y='Expected Return (%)',
        color='Sharpe Ratio',
        color_continuous_scale='plasma',
        opacity=0.85,
        hover_data={'Volatility (%)': ':.2f', 'Expected Return (%)': ':.2f', 'Sharpe Ratio': ':.2f'},
        title="Efficient Frontier with Optimal Complete Portfolio"
    )

    fig.add_trace(go.Scatter(
        x=[0],
        y=[rf * 100],
        mode='markers+text',
        name=f"Risk-Free Rate ({rf*100:.2f}%)",
        marker=dict(color='deepskyblue', size=14),
        text=["Risk-Free"],
        textposition="bottom left",
        hovertemplate="Risk-Free Rate: %{y:.2f}%<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=[sigma_p * 100],
        y=[mu_p * 100],
        mode='markers+text',
        name=f"Optimal Risky Portfolio ({', '.join(tickers)})",
        marker=dict(color='red', size=16, symbol='star'),
        text=["Optimal Risky"],
        textposition="top left",
        hovertemplate="Optimal Risky Portfolio:<br>Volatility: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=[sigma_c * 100],
        y=[expected_rc * 100],
        mode='markers+text',
        name=f"Optimal Complete Portfolio (y={y_capped:.2f})",
        marker=dict(color='lime', size=16, symbol='diamond'),
        text=["Complete Portfolio"],
        textposition="bottom right",
        hovertemplate="Complete Portfolio:<br>Volatility: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>"
    ))

    if abs(y_opt - y_capped) > 0.01:
        sigma_leverage = y_opt * sigma_p
        rc_leverage = y_opt * mu_p + (1 - y_opt) * rf
        fig.add_trace(go.Scatter(
            x=[sigma_leverage * 100],
            y=[rc_leverage * 100],
            mode='markers+text',
            name=f"Leveraged Portfolio (y={y_opt:.2f})",
            marker=dict(color='magenta', size=16, symbol='x'),
            text=["Leveraged"],
            textposition="bottom left",
            hovertemplate="Leveraged Portfolio:<br>Volatility: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>"
        ))

    slope = (mu_p - rf) / sigma_p if sigma_p > 0 else 0
    max_sigma = df_sim['Volatility (%)'].max() if not df_sim.empty else sigma_p * 150
    x_line = np.linspace(0, max_sigma * 1.1, 300)
    y_line = rf * 100 + slope * x_line

    fig.add_trace(go.Scatter(
        x=x_line,
        y=y_line,
        mode='lines',
        name="Capital Allocation Line (CAL)",
        line=dict(color='red', dash='dash', width=2),
        hoverinfo='skip'
    ))

    fig.update_layout(
        height=650,
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        font=dict(color='white'),
        title_x=0.25,
        title_y=0.98,
        title_font=dict(size=20, family='Arial, sans-serif', color='white'),
        legend=dict(
            bgcolor='#1e1e1e',
            font=dict(color='white'),
            yanchor='top',
            y=0.98,
            xanchor='left',
            x=0.02
        ),
        margin=dict(t=80, b=40, l=60, r=40)
    )

    fig.update_xaxes(title='Volatility (%)', color='white', showgrid=False)
    fig.update_yaxes(title='Expected Return (%)', color='white', showgrid=False)

    st.plotly_chart(fig, use_container_width=True)
