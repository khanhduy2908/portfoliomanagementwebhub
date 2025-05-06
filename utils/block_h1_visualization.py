import streamlit as st

def display_portfolio_info(portfolio_info, alloc_df):
    st.subheader("Optimal Complete Portfolio Summary")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Portfolio**: `{portfolio_info['Portfolio Name']}`")
        st.markdown(f"**Risk Aversion (A)**: `{best_portfolio['Expected Return (%)']}`")
        st.markdown(f"**y***: `{portfolio_info['y_opt']:.4f}`")
        st.markdown(f"**y_capped**: `{portfolio_info['y_capped']:.4f}`")

    with col2:
        st.markdown(f"**Expected Return (E(rc))**: `{portfolio_info['Expected Return']:.4f}`")
        st.markdown(f"**Volatility (σ_c)**: `{portfolio_info['Portfolio Volatility']:.4f}`")
        st.markdown(f"**Risk-Free Rate (r_f)**: `{portfolio_info['Risk-Free Rate']:.4f}`")
        st.markdown(f"**Utility (U)**: `{portfolio_info['Utility']:.4f}`")

    with col3:
        st.markdown(f"**Total Capital**: `{portfolio_info['Total Capital']:,.0f} VND`")
        st.markdown(f"**Risky Capital**: `{portfolio_info['Capital Risky']:,.0f} VND`")
        st.markdown(f"**Risk-Free Capital**: `{portfolio_info['Capital Risk-Free']:,.0f} VND`")

    # --- Capital Allocation Table ---
    st.markdown("---")
    st.markdown("### Capital Allocation to Risky Assets")
    st.dataframe(alloc_df.style.format({"Allocated Capital (VND)": "{:,.0f}"}), use_container_width=True)
