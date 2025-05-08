import streamlit as st
import pandas as pd

def display_portfolio_info(portfolio_info, alloc_df):
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

    st.markdown("### Risk Profile Commentary")
    if portfolio_info['risk_score'] <= 17:
        st.info("🛡️ This is a **conservative portfolio** with high exposure to risk-free assets, focusing on capital preservation.")
    elif 18 <= portfolio_info['risk_score'] <= 27:
        st.info("⚖️ This is a **balanced portfolio**, blending growth and stability through diversified asset exposure.")
    else:
        st.info("🚀 This is an **aggressive portfolio** aiming for high returns, with minimal allocation to risk-free assets.")
