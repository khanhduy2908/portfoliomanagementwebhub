import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

def run(portfolio_info: dict, capital_alloc: dict, tickers: list, allocation_matrix: dict, risk_level: str, time_horizon: str):
    st.markdown("### Asset Allocation Overview")

    if not capital_alloc or not tickers:
        st.warning("⚠️ Missing capital allocation or tickers.")
        return

    # Lấy dữ liệu chuẩn từ portfolio_info
    capital_cash = portfolio_info['capital_cash']
    capital_bond = portfolio_info['capital_bond']
    capital_stock = portfolio_info['capital_stock']
    capital_risky = portfolio_info['capital_risky']

    # Tính tổng vốn cổ phiếu từ capital_alloc (đồng bộ với portfolio_info)
    capital_from_stocks = sum([capital_alloc.get(t, 0) for t in tickers])
    if abs(capital_risky - capital_from_stocks) > 1:
        capital_from_stocks = capital_risky  # Ưu tiên số từ block H

    # Tổng vốn thực tế dùng để tính %
    total = capital_cash + capital_bond + capital_from_stocks
    if total <= 0:
        st.error("⚠️ Total capital is zero. Cannot compute allocation.")
        return

    # Kích thước và nhãn cho biểu đồ pie
    sizes = [capital_cash, capital_bond] + [capital_alloc.get(t, 0) for t in tickers]
    labels = ['Cash', 'Bond'] + tickers

    # Vẽ biểu đồ Pie
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

    # Bảng chi tiết vốn phân bổ
    with col2:
        stock_total = capital_from_stocks
        summary_df = pd.DataFrame({
            "Asset Class / Ticker": ['Cash', 'Bond'] + tickers + ['Stock Total', 'Total'],
            "Capital (VND)": (
                [f"{capital_cash:,.0f}", f"{capital_bond:,.0f}"] +
                [f"{capital_alloc.get(t, 0):,.0f}" for t in tickers] +
                [f"{stock_total:,.0f}", f"{total:,.0f}"]
            ),
            "Allocation (%)": (
                [f"{capital_cash / total * 100:.1f}%", f"{capital_bond / total * 100:.1f}%"] +
                [f"{capital_alloc.get(t, 0) / total * 100:.1f}%" for t in tickers] +
                [f"{stock_total / total * 100:.1f}%", "100.0%"]
            )
        })
        st.markdown("### Allocation Table")
        st.dataframe(summary_df, use_container_width=True, height=400)

    # Bảng Target vs Actual Allocation dưới biểu đồ
    st.markdown("### Target vs Actual Allocation Comparison")

    target_allocation = allocation_matrix.get((risk_level, time_horizon), {
        "cash": portfolio_info['target_cash_ratio'],
        "bond": portfolio_info['target_bond_ratio'],
        "stock": portfolio_info['target_stock_ratio']
    })

    target_ratios = {
        "Cash": target_allocation['cash'],
        "Bonds": target_allocation['bond'],
        "Stocks": target_allocation['stock']
    }

    actual_ratios = {
        "Cash": capital_cash / total,
        "Bonds": capital_bond / total,
        "Stocks": stock_total / total
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

    st.dataframe(df_compare, use_container_width=True, height=200)

    # Cảnh báo nếu lệch lớn hơn 5%
    large_deviation = [
        k for k in ["Cash", "Bonds", "Stocks"]
        if abs(actual_ratios[k] - target_ratios[k]) > 0.05
    ]

    if large_deviation:
        st.warning(f"⚠️ Allocation deviates significantly (>5%) for: {', '.join(large_deviation)}")
