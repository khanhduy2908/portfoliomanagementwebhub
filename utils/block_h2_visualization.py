import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

def run(portfolio_info: dict, capital_alloc: dict, tickers: list):
    st.markdown("### Asset Allocation Overview")

    if not capital_alloc or not tickers:
        st.warning("⚠️ Missing capital allocation or tickers.")
        return

    # --- Lấy số liệu vốn từ portfolio_info để đảm bảo đồng bộ ---
    capital_cash = portfolio_info.get('capital_cash', 0)
    capital_bond = portfolio_info.get('capital_bond', 0)
    capital_stock = portfolio_info.get('capital_stock', 0)
    capital_risky = portfolio_info.get('capital_risky', 0)

    # Tổng vốn cổ phiếu thực tế từ capital_alloc
    capital_from_stocks = sum([capital_alloc.get(t, 0) for t in tickers])
    # Đồng bộ nếu lệch lớn hơn 1 VND (rất nhỏ)
    if abs(capital_risky - capital_from_stocks) > 1:
        capital_from_stocks = capital_risky

    # Tổng vốn danh mục đầu tư
    sizes = [capital_cash, capital_bond] + [capital_alloc.get(t, 0) for t in tickers]
    labels = ['Cash', 'Bond'] + tickers
    total = sum(sizes)

    if total <= 0:
        st.error("⚠️ Total capital is zero. Cannot compute allocation.")
        return

    # --- Vẽ biểu đồ tròn phân bổ ---
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

    # --- Bảng dữ liệu chi tiết ---
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
                [f"{capital_cash/total*100:.1f}%", f"{capital_bond/total*100:.1f}%"] +
                [f"{capital_alloc.get(t, 0)/total*100:.1f}%" for t in tickers] +
                [f"{stock_total/total*100:.1f}%", "100.0%"]
            )
        })

        st.markdown("### Allocation Table")
        st.dataframe(summary_df, use_container_width=True, height=400)

    # --- Cảnh báo nếu tỷ trọng cổ phiếu thực tế lệch so với mục tiêu quá 5% ---
    target_stock_ratio = portfolio_info.get("target_stock_ratio", None)
    actual_stock_ratio = stock_total / total

    if target_stock_ratio is not None:
        diff = abs(actual_stock_ratio - target_stock_ratio)
        if diff > 0.05:
            st.warning(f"⚠️ Stock allocation ({actual_stock_ratio*100:.1f}%) deviates from target ({target_stock_ratio*100:.1f}%) by more than 5%.")
