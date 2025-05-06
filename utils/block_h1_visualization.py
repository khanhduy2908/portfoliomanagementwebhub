# utils/block_h1_visualization.py

import streamlit as st

def display_portfolio_info(portfolio_info, alloc_df):
    st.subheader("Complete Portfolio Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Portfolio Name:** `{portfolio_info['portfolio_name']}`")
        st.markdown(f"**Expected Return (E(rc)) [%]:** `{portfolio_info['expected_rc'] * 100:.2f}`")
        st.markdown(f"**Portfolio Risk (σ_c) [%]:** `{portfolio_info['sigma_c'] * 100:.2f}`")
        st.markdown(f"**Risk-Free Capital (VND):** `{portfolio_info['capital_rf']:,.0f}`")
        st.markdown(f"**Risky Capital (VND):** `{portfolio_info['capital_risky']:,.0f}`")

    with col2:
        st.markdown(f"**Risk Aversion Coefficient (A):** `{portfolio_info['A']}`")
        st.markdown(f"**Optimal y* (risky allocation):** `{portfolio_info['y_opt']:.4f}`")
        st.markdown(f"**Capped y (used):** `{portfolio_info['y_capped']:.4f}`")
        st.markdown(f"**Sharpe Utility Value:** `{portfolio_info['utility']:.4f}`")
        st.markdown(f"**Risk-Free Rate [%]:** `{portfolio_info['rf'] * 100:.2f}`")

    st.markdown("### Capital Allocation to Risky Assets")
    st.dataframe(
        alloc_df.style.format({"Allocated Capital (VND)": "{:,.0f}"}),
        use_container_width=True
    )
