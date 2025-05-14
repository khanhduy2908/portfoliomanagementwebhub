import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

def run(portfolio_info: dict, capital_alloc: dict, tickers: list, allocation_matrix: dict, risk_level: str, time_horizon: str):
    st.markdown("### Asset Allocation Overview")

    if not capital_alloc or not tickers:
        st.warning("⚠️ Missing capital allocation or tickers.")
        return

    # Lấy vốn cash, bond từ portfolio_info (nguồn chuẩn)
    capital_cash = portfolio_info.get('capital_cash', 0)
    capital_bond = portfolio_info.get('capital_bond', 0)

    # Lấy vốn cổ phiếu từng ticker từ capital_alloc
    capital_tickers = [capital_alloc.get(t, 0) for t in tickers]

    # Tổng vốn cổ phiếu tính từ capital_alloc để chuẩn bị biểu đồ và bảng
    capital_stock = sum(capital_tickers)

    # Chuẩn bị dữ liệu biểu đồ pie chart
    sizes = [capital_cash, capital_bond] + capital_tickers
    labels = ['Cash', 'Bond'] + tickers

    total = sum(sizes)
    if total <= 0:
        st.error("⚠️ Total capital is zero. Cannot compute allocation.")
        return

    # Vẽ biểu đồ Pie và bảng chi tiết bên cạnh
    col1, col2 = st.columns([2, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(6, 5), facecolor='#1e1e1e')
        cmap = plt.cm.get_cmap('tab20', len(labels))
        colors = [cmap(i) for i in range(len(labels))]

        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            textprops={'color': 'white', 'fontsize': 9}
        )
        for text in texts:
            text.set_color('white')
        for autotext in autotexts:
            autotext.set_color('black')

        ax.set_title("Capital Allocation by Asset Class and Ticker", fontsize=13, color='white')
        ax.set_facecolor('#1e1e1e')
        fig.patch.set_facecolor('#1e1e1e')
        fig.tight_layout()
        st.pyplot(fig)

    with col2:
        summary_df = pd.DataFrame({
            "Asset Class / Ticker": ['Cash', 'Bond'] + tickers + ['Stock Total', 'Total'],
            "Capital (VND)": (
                [f"{capital_cash:,.0f}", f"{capital_bond:,.0f}"] +
                [f"{amt:,.0f}" for amt in capital_tickers] +
                [f"{capital_stock:,.0f}", f"{total:,.0f}"]
            ),
            "Allocation (%)": (
                [f"{capital_cash/total*100:.1f}%", f"{capital_bond/total*100:.1f}%"] +
                [f"{amt/total*100:.1f}%" for amt in capital_tickers] +
                [f"{capital_stock/total*100:.1f}%", "100.0%"]
            )
        })
        st.markdown("### Allocation Table")
        st.dataframe(summary_df, use_container_width=True, height=400)

    # Bảng Target vs Actual Allocation (Actual lấy đúng từ biểu đồ trên)
    st.markdown("### Target vs Actual Allocation Comparison")

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
            "Target Ratio": f"{target_ratios[k]*100:.1f}%",
            "Actual Ratio": f"{actual_ratios[k]*100:.1f}%",
            "Difference": f"{(actual_ratios[k] - target_ratios[k])*100:.1f}%"
        }
        for k in ["Cash", "Bonds", "Stocks"]
    ])

    st.dataframe(df_compare, use_container_width=True, height=250)

    # Cảnh báo lệch tỷ trọng lớn hơn 5%
    large_deviation = [
        k for k in ["Cash", "Bonds", "Stocks"]
        if abs(actual_ratios[k] - target_ratios[k]) > 0.05
    ]

    if large_deviation:
        st.warning(f"⚠️ Allocation deviates significantly (>5%) for: {', '.join(large_deviation)}")
