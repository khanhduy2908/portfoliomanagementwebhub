# utils/block_h1_visualization.py

import streamlit as st
import pandas as pd

def display_portfolio_info(portfolio_info, alloc_df):
    st.markdown("### Optimal Complete Portfolio Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Selected Portfolio**: `{portfolio_info['portfolio_name']}`")
        st.markdown(f"**Risk Tolerance Score (Mapped A)**: {portfolio_info['A']}")
        st.markdown(f"**Expected Return**: `{portfolio_info['expected_rc'] * 100:.2f}%` per month")
        st.markdown(f"**Portfolio Volatility**: `{portfolio_info['sigma_c'] * 100:.2f}%` per month")
        st.markdown(f"**Portfolio Utility Score**: `{portfolio_info['utility']:.2f}`")

    with col2:
        st.markdown(f"**Optimal Risk Exposure (y\*)**: `{portfolio_info['y_opt'] * 100:.1f}%`")
        st.markdown(f"**Final Exposure Used**: `{portfolio_info['y_capped'] * 100:.1f}%`")
        st.markdown(f"**Capital in Risk-Free Asset**: `{portfolio_info['capital_rf']:,.0f} VND`")
        st.markdown(f"**Capital in Risky Assets**: `{portfolio_info['capital_risky']:,.0f} VND`")
        st.markdown(f"**Total Capital**: `{portfolio_info['capital_rf'] + portfolio_info['capital_risky']:,.0f} VND`")

    st.markdown("---")
    st.markdown("### Allocation to Risky Assets")
    st.dataframe(
        alloc_df.style.format({"Allocated Capital (VND)": "{:,.0f}"}),
        use_container_width=True
    )
