# utils/block_h2_visualization.py

import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

def run(capital_alloc: dict, capital_cash: float, capital_bond: float, capital_stock: float, tickers: list):
    st.markdown("### Asset Allocation Overview")

    if not capital_alloc or not tickers:
        st.warning("⚠️ Missing capital allocation or tickers.")
        return

    # === Tính tổng vốn cổ phiếu thực tế ===
    capital_from_stocks = sum([capital_alloc.get(t, 0) for t in tickers])
    if abs(capital_stock - capital_from_stocks) > 1:
        capital_stock = capital_from_stocks

    # === Tổng vốn đầu tư ===
    sizes = [capital_cash, capital_bond] + [capital_alloc.get(t, 0) for t in tickers]
    labels = ['Cash', 'Bond'] + tickers
    total = sum(sizes)

    if total <= 0:
        st.error("⚠️ Total capital is zero. Cannot compute allocation.")
        return

    # === Tính phần trăm phân bổ ===
    percentages = [s / total * 100 for s in sizes]

    # === Biểu đồ và bảng ===
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
        stock_total = sum([capital_alloc.get(t, 0) for t in tickers])
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

    # --- Cảnh báo nếu tổng stock allocation lệch hơn 5% ---
    target_stock_ratio = st.session_state.get("target_stock_ratio", None)
    actual_stock_ratio = stock_total / total

    if target_stock_ratio is not None:
        diff = abs(actual_stock_ratio - target_stock_ratio)
        if diff > 0.05:
            st.warning(f"⚠️ Stock allocation ({actual_stock_ratio*100:.1f}%) deviates from target ({target_stock_ratio*100:.1f}%) by more than 5%.")
