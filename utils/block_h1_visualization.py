import streamlit as st
import pandas as pd

def display_portfolio_info(portfolio_info):
    st.markdown("### Optimal Complete Portfolio Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Portfolio Name:** `{portfolio_info['portfolio_name']}`")
        st.markdown(f"**Risk Score (User Input):** `{portfolio_info['risk_score']}`")
        st.markdown(f"**Mapped Risk Aversion (A):** `{portfolio_info['A']}`")
        st.markdown(f"**Expected Monthly Return:** `{portfolio_info['expected_rc'] * 100:.2f}%`")
        st.markdown(f"**Monthly Volatility:** `{portfolio_info['sigma_c'] * 100:.2f}%`")
        st.markdown(f"**Portfolio Utility Score:** `{portfolio_info['utility']:.2f}`")

    with col2:
        st.markdown(f"**Optimal Risk Exposure (y\*):** `{portfolio_info['y_opt'] * 100:.1f}%`")
        st.markdown(f"**Final Risk Exposure Used:** `{portfolio_info['y_capped'] * 100:.1f}%`")
        st.markdown(f"**Max Risk-Free Ratio Allowed:** `{portfolio_info['max_rf_ratio'] * 100:.0f}%`")
        st.markdown(f"**Capital in Risk-Free Asset:** `{portfolio_info['capital_rf']:,.0f} VND`")
        st.markdown(f"**Capital in Risky Assets:** `{portfolio_info['capital_risky']:,.0f} VND`")
        st.markdown(f"**Total Capital Allocated:** `{portfolio_info['capital_rf'] + portfolio_info['capital_risky']:,.0f} VND`")

