# utils/block_h2_visualization.py

import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

def run(capital_alloc: dict, capital_cash: float, capital_bond: float, capital_stock: float, tickers: list):
    st.markdown("### Asset Allocation by Class and Ticker")

    if not capital_alloc or not tickers:
        st.warning("⚠️ Missing capital allocation or tickers.")
        return

    # 1. Tính lại vốn cổ phiếu nếu có sai lệch
    capital_from_alloc = sum([capital_alloc.get(t, 0) for t in tickers])
    if abs(capital_stock - capital_from_alloc) > 1:
        capital_stock = capital_from_alloc

    # 2. Chuẩn hóa tên lớp tài sản
    labels = ['Cash', 'Bond'] + tickers
    sizes = [capital_cash, capital_bond] + [capital_alloc.get(t, 0) for t in tickers]
    total = sum(sizes)

    if total <= 0:
        st.error("⚠️ Total capital is zero. Cannot render allocation.")
        return

    percentages = [s / total * 100 for s in sizes]

    # 3. Hiển thị
    col1, col2 = st.columns([2, 1])

    # Biểu đồ Pie
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

        ax.set_title("Capital Distribution", fontsize=13, color='white')
        ax.set_facecolor('#1e1e1e')
        fig.patch.set_facecolor('#1e1e1e')
        st.pyplot(fig)

    # Bảng chi tiết
    with col2:
        summary_df = pd.DataFrame({
            "Asset Class / Ticker": labels,
            "Capital (VND)": [f"{v:,.0f}" for v in sizes],
            "Allocation (%)": [f"{p:.1f}%" for p in percentages]
        })
        total_row = pd.DataFrame([{
            "Asset Class / Ticker": "Total",
            "Capital (VND)": f"{total:,.0f}",
            "Allocation (%)": "100.0%"
        }])
        summary_df = pd.concat([summary_df, total_row], ignore_index=True)

        st.markdown("### Allocation Table")
        st.dataframe(summary_df, use_container_width=True, height=300)
