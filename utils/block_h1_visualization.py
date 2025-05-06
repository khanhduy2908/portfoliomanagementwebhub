# utils/block_h1_visualization.py

import streamlit as st
import pandas as pd

def display_portfolio_info(portfolio_info, alloc_df):
    st.subheader("Optimal Complete Portfolio Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Portfolio Name**: {portfolio_info['portfolio_name']}")
        st.markdown(f"**Risk Aversion (A)**: {portfolio_info['A']}")
        st.markdown(f"**Expected Return (E(rc))**: {portfolio_info['expected_rc']:.4f}")
        st.markdown(f"**Portfolio Risk (σ_c)**: {portfolio_info['sigma_c']:.4f}")
        st.markdown(f"**Utility (U)**: {portfolio_info['utility']:.4f}")

    with col2:
        st.markdown(f"**Risk Exposure (y*)**: {portfolio_info['y_opt']:.4f}")
        st.markdown(f"**Capped Exposure (y_capped)**: {portfolio_info['y_capped']:.4f}")
        st.markdown(f"**Risk-Free Capital**: {portfolio_info['capital_rf']:,.0f} VND")
        st.markdown(f"**Risky Capital**: {portfolio_info['capital_risky']:,.0f} VND")
        st.markdown(f"**Total Capital**: {portfolio_info['capital_rf'] + portfolio_info['capital_risky']:,.0f} VND")

    st.markdown("### Capital Allocation to Risky Assets")
    st.dataframe(
        alloc_df.style.format({"Allocated Capital (VND)": "{:,.0f}"}),
        use_container_width=True
    )
