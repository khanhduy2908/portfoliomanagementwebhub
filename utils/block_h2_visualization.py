import pandas as pd
import plotly.express as px
import streamlit as st

def run(portfolio_info: dict, capital_alloc: dict, tickers: list, allocation_matrix: dict, risk_level: str, time_horizon: str):
    st.subheader("Asset Allocation Overview")

    if not capital_alloc or not tickers:
        st.warning("⚠️ Missing capital allocation or tickers.")
        return

    capital_cash = portfolio_info.get('capital_cash', 0)
    capital_bond = portfolio_info.get('capital_bond', 0)
    capital_tickers = [capital_alloc.get(t, 0) for t in tickers]
    capital_stock = sum(capital_tickers)
    sizes = [capital_cash, capital_bond] + capital_tickers
    labels = ['Cash', 'Bond'] + tickers
    total = sum(sizes)

    if total <= 0:
        st.error("⚠️ Total capital is zero. Cannot compute allocation.")
        return

    pie_df = pd.DataFrame({
        'Asset Class / Ticker': labels,
        'Capital (VND)': sizes,
        'Allocation (%)': [v / total * 100 for v in sizes]
    })

    col1, col2 = st.columns([1.6, 1])

    with col1:
        fig = px.pie(
            pie_df,
            names='Asset Class / Ticker',
            values='Allocation (%)',
            hole=0.35,
            title="Capital Allocation by Asset Class and Ticker"
        )
        fig.update_traces(
            textinfo='percent+label',
            textfont_size=14,
            hovertemplate='%{label}: %{percent:.2f}<extra></extra>'
        )
        fig.update_layout(
            title_x=0.25,  # Căn giữa title ngang
            title_y=0.95,
            plot_bgcolor='#1e1e1e',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white', size=14),
            margin=dict(t=70, b=20, l=20, r=20),
            legend=dict(
                orientation="v",
                y=0.5,
                x=1.05,
                font=dict(size=12),
                bgcolor='rgba(0,0,0,0)',
                bordercolor='rgba(255,255,255,0.2)',
                borderwidth=1
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
            <div style="color: white; font-size: 13px; margin-top: 15px;">
                <b>Note:</b> Capital allocation includes cash, bonds, and individual stocks.<br>
                Allocation percentages reflect the portion of total capital assigned to each asset.<br>
                The pie chart visualizes diversification and portfolio balance.
            </div>
            """, unsafe_allow_html=True
        )

    with col2:
        summary_df = pd.DataFrame({
            "Asset Class / Ticker": labels + ['Stock Total', 'Total'],
            "Capital (VND)": [f"{v:,.0f}" for v in sizes] + [f"{capital_stock:,.0f}", f"{total:,.0f}"],
            "Allocation (%)": [f"{v / total * 100:.1f}%" for v in sizes] + [f"{capital_stock / total * 100:.1f}%", "100.0%"]
        })
        st.markdown("#### Allocation Table")
        st.dataframe(summary_df, use_container_width=True, height=410)

    # Target vs Actual Allocation Comparison
    st.markdown("#### Target vs Actual Allocation Comparison")

    target_allocation = allocation_matrix.get((risk_level, time_horizon), {
        "cash": portfolio_info.get('target_cash_ratio', 0),
        "bond": portfolio_info.get('target_bond_ratio', 0),
        "stock": portfolio_info.get('target_stock_ratio', 0)
    })

    target_ratios = {
        "Cash": target_allocation['cash'],
        "Bonds": target_allocation['bond'],
        "Stocks": target_allocation['stock']
    }

    actual_ratios = {
        "Cash": capital_cash / total,
        "Bonds": capital_bond / total,
        "Stocks": capital_stock / total
    }

    df_compare = pd.DataFrame([
        {
            "Asset Class": k,
            "Target Ratio": f"{target_ratios[k] * 100:.1f}%",
            "Actual Ratio": f"{actual_ratios[k] * 100:.1f}%",
            "Difference": f"{(actual_ratios[k] - target_ratios[k]) * 100:.1f}%"
        }
        for k in ["Cash", "Bonds", "Stocks"]
    ])
    st.dataframe(df_compare, use_container_width=True, height=250)

    large_deviation = [
        k for k in ["Cash", "Bonds", "Stocks"]
        if abs(actual_ratios[k] - target_ratios[k]) > 0.05
    ]
    if large_deviation:
        st.warning(f"⚠️ Allocation deviates significantly (>5%) for: {', '.join(large_deviation)}")
