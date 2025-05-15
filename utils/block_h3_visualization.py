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

    # --- Validate inputs and extract portfolio info ---
    try:
        best_key = list(best_portfolio.keys())[0]
        tickers = list(best_key)
        result = best_portfolio[best_key]
        weights = np.array([result['Weights'][t] for t in tickers])
    except Exception as e:
        st.error(f"❌ Failed to extract best portfolio info: {e}")
        return

    # --- Retrieve realistic expected returns and covariance matrix ---
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

    # --- Simulate Efficient Frontier for smooth visualization ---
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

    # --- Build scatter plot ---
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

    # Add points with labels positioned to avoid overlap
    annotations_pos = {
        "Risk-Free": dict(x=0.5, y=rf*100 + 0.1, xref="x", yref="y", showarrow=False, font=dict(color="deepskyblue")),
        "Optimal Risky": dict(x=sigma_p*100 + 5, y=mu_p*100 + 0.1, xref="x", yref="y", showarrow=False, font=dict(color="red")),
        "Complete Portfolio": dict(x=sigma_c*100 + 5, y=expected_rc*100 + 0.1, xref="x", yref="y", showarrow=False, font=dict(color="lime")),
        "Leveraged": dict(x=(y_opt*sigma_p)*100 + 5, y=(y_opt*mu_p + (1-y_opt)*rf)*100 + 0.1, xref="x", yref="y", showarrow=False, font=dict(color="magenta"))
    }

    fig.add_trace(go.Scatter(
        x=[0],
        y=[rf * 100],
        mode='markers',
        marker=dict(color='deepskyblue', size=14),
        name=f"Risk-Free Rate ({rf*100:.2f}%)",
        hovertemplate="Risk-Free Rate: %{y:.2f}%<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=[sigma_p * 100],
        y=[mu_p * 100],
        mode='markers',
        marker=dict(color='red', size=16, symbol='star'),
        name=f"Optimal Risky Portfolio ({', '.join(tickers)})",
        hovertemplate="Optimal Risky Portfolio:<br>Volatility: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=[sigma_c * 100],
        y=[expected_rc * 100],
        mode='markers',
        marker=dict(color='lime', size=16, symbol='diamond'),
        name=f"Optimal Complete Portfolio (y={y_capped:.2f})",
        hovertemplate="Complete Portfolio:<br>Volatility: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>"
    ))

    if abs(y_opt - y_capped) > 0.01:
        sigma_leverage = y_opt * sigma_p
        rc_leverage = y_opt * mu_p + (1 - y_opt) * rf
        fig.add_trace(go.Scatter(
            x=[sigma_leverage * 100],
            y=[rc_leverage * 100],
            mode='markers',
            marker=dict(color='magenta', size=16, symbol='x'),
            name=f"Leveraged Portfolio (y={y_opt:.2f})",
            hovertemplate="Leveraged Portfolio:<br>Volatility: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>"
        ))

    # Add annotation labels on top-left corners of points
    for label, ann in annotations_pos.items():
        if label == "Leveraged" and abs(y_opt - y_capped) <= 0.01:
            continue
        fig.add_annotation(text=label, **ann)

    # Capital Allocation Line (CAL)
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

    # Layout with dark theme and grid hidden
    fig.update_layout(
        height=650,
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        font=dict(color='white'),
        title_x=0.5,
        legend=dict(bgcolor='#1e1e1e', font=dict(color='white')),
        margin=dict(t=70, b=50, l=70, r=40)
    )
    fig.update_xaxes(title='Volatility (%)', color='white', showgrid=False)
    fig.update_yaxes(title='Expected Return (%)', color='white', showgrid=False)

    st.plotly_chart(fig, use_container_width=True)
