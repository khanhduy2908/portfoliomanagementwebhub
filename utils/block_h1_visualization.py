import streamlit as st
import pandas as pd
import numpy as np

def display_portfolio_info(portfolio_info: dict):
    st.markdown("### Optimal Complete Portfolio Summary")

    col1, col2 = st.columns(2)

    # --- Left Column: Risk Profile & Return Stats ---
    with col1:
        st.markdown(f"**Portfolio Name:** `{portfolio_info['portfolio_name']}`")
        st.markdown(f"**Risk Tolerance Score:** `{portfolio_info['risk_score']}`")
        st.markdown(f"**Risk Aversion Coefficient (A):** `{portfolio_info['A']:.2f}`")
        st.markdown(f"**Expected Monthly Return:** `{portfolio_info['expected_rc'] * 100:.2f}%`")
        st.markdown(f"**Monthly Volatility:** `{portfolio_info['sigma_c'] * 100:.2f}%`")
        st.markdown(f"**Utility Score:** `{portfolio_info['utility']:.2f}`")

    # --- Right Column: Capital Structure ---
    with col2:
        y_opt = portfolio_info['y_opt']
        y_capped = portfolio_info['y_capped']
        y_diff = y_opt - y_capped

        st.markdown(f"**y\* (Optimal Risk Exposure):** `{y_opt * 100:.1f}%`")
        if y_diff > 0.005:
            st.markdown(f"**y (Final Used):** `{y_capped * 100:.1f}%` ⚠️ _adjusted due to constraints_")
        else:
            st.markdown(f"**y (Final Used):** `{y_capped * 100:.1f}%`")

        st.markdown(f"**Max Risk-Free Ratio:** `{portfolio_info['max_rf_ratio'] * 100:.0f}%`")
        st.markdown(f"**Capital in Risk-Free Assets:** `{portfolio_info['capital_rf']:,.0f} VND`")
        st.markdown(f"**Capital in Risky Assets (Equity):** `{portfolio_info['capital_risky']:,.0f} VND`")
        st.markdown(f"**Total Capital:** `{portfolio_info['capital_rf'] + portfolio_info['capital_risky']:,.0f} VND`")

    # --- Risk-Free Allocation Limit Warning ---
    rf_limit = portfolio_info['max_rf_ratio'] * (portfolio_info['capital_rf'] + portfolio_info['capital_risky'])
    if portfolio_info['capital_rf'] > rf_limit:
        st.warning(f"⚠️ Risk-Free allocation exceeds maximum cap ({portfolio_info['max_rf_ratio']*100:.0f}%)")

    # --- Allocation Comparison ---
    st.markdown("### Target vs Actual Allocation Comparison")

    total_cap = (
        portfolio_info['capital_cash'] +
        portfolio_info['capital_bond'] +
        portfolio_info['capital_stock']
    )

    target_ratios = {
        "Cash": portfolio_info['target_cash_ratio'],
        "Bonds": portfolio_info['target_bond_ratio'],
        "Stocks": portfolio_info['target_stock_ratio']
    }

    actual_ratios = {
        "Cash": portfolio_info['actual_cash_ratio'],
        "Bonds": portfolio_info['actual_bond_ratio'],
        "Stocks": portfolio_info['actual_stock_ratio']
    }

    key_map = {
        "Cash": "capital_cash",
        "Bonds": "capital_bond",
        "Stocks": "capital_stock"
    }

    df_compare = pd.DataFrame([
        {
            "Asset Class": k,
            "Target Ratio": f"{target_ratios[k]*100:.1f}%",
            "Actual Ratio": f"{actual_ratios[k]*100:.1f}%",
            "Capital (VND)": f"{portfolio_info[key_map[k]]:,.0f}",
            "Difference": f"{(actual_ratios[k] - target_ratios[k]) * 100:.1f}%"
        }
        for k in ["Cash", "Bonds", "Stocks"]
    ])

    st.dataframe(df_compare, use_container_width=True, hide_index=True)

    # --- Warning if deviation > 5% absolute ---
    large_deviation = [
        k for k in ["Cash", "Bonds", "Stocks"]
        if abs(actual_ratios[k] - target_ratios[k]) > 0.05
    ]

    if large_deviation:
        st.error(f"⚠️ Allocation deviates significantly (>5%) for: {', '.join(large_deviation)}")
