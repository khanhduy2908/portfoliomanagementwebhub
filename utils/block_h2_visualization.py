import streamlit as st

def display_portfolio_info(portfolio_info: dict, risk_level: str, time_horizon: str):
    st.markdown("### Optimal Complete Portfolio Summary")

    # --- Left Column: Risk Profile & Return Stats ---
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Portfolio Name:** `{portfolio_info['portfolio_name']}`")
        st.markdown(f"**Risk Tolerance Score:** `{portfolio_info['risk_score']}`")
        st.markdown(f"**Risk Aversion Coefficient (A):** `{portfolio_info['A']:.2f}`")
        st.markdown(f"**Expected Monthly Return:** `{portfolio_info['expected_rc'] * 100:.2f}%`")
        st.markdown(f"**Monthly Volatility:** `{portfolio_info['sigma_c'] * 100:.2f}%`")
        st.markdown(f"**Utility Score:** `{portfolio_info['utility']:.2f}`")

    with col2:
        y_opt = portfolio_info['y_opt']
        y_capped = portfolio_info['y_capped']
        y_diff = y_opt - y_capped

        st.markdown(f"**y* (Optimal Risk Exposure):** `{y_opt * 100:.1f}%`")
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
