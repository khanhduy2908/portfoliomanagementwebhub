import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

def run(portfolio_info: dict, capital_alloc: dict, tickers: list):
    st.markdown("### Asset Allocation Overview")

    if not capital_alloc or not tickers:
        st.warning("⚠️ Missing capital allocation or tickers.")
        return

    # Lấy dữ liệu vốn từ portfolio_info (đảm bảo đồng bộ với block H)
    capital_cash = portfolio_info['capital_cash']
    capital_bond = portfolio_info['capital_bond']
    capital_stock = portfolio_info['capital_stock']
    capital_risky = portfolio_info['capital_risky']

    # Tổng vốn cổ phiếu thực tế từ chi tiết phân bổ cổ phiếu
    capital_from_stocks = sum([capital_alloc.get(t, 0) for t in tickers])
    if abs(capital_risky - capital_from_stocks) > 1:
        capital_from_stocks = capital_risky  # Ưu tiên dữ liệu từ block H

    # Tổng vốn toàn danh mục
    sizes = [capital_cash, capital_bond] + [capital_alloc.get(t, 0) for t in tickers]
    labels = ['Cash', 'Bond'] + tickers
    total = sum(sizes)

    if total <= 0:
        st.error("⚠️ Total capital is zero. Cannot compute allocation.")
        return

    # Biểu đồ tròn phân bổ vốn
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

    # Bảng so sánh Target vs Actual Allocation
    with col2:
        target_cash = portfolio_info.get('target_cash_ratio', 0)
        target_bond = portfolio_info.get('target_bond_ratio', 0)
        target_stock = portfolio_info.get('target_stock_ratio', 0)

        actual_cash = portfolio_info.get('actual_cash_ratio', capital_cash / total)
        actual_bond = portfolio_info.get('actual_bond_ratio', capital_bond / total)
        actual_stock = portfolio_info.get('actual_stock_ratio', capital_stock / total)

        df_compare = pd.DataFrame({
            "Asset Class": ["Cash", "Bonds", "Stocks"],
            "Target Ratio": [f"{target_cash*100:.1f}%", f"{target_bond*100:.1f}%", f"{target_stock*100:.1f}%"],
            "Actual Ratio": [f"{actual_cash*100:.1f}%", f"{actual_bond*100:.1f}%", f"{actual_stock*100:.1f}%"],
            "Difference": [f"{(actual_cash - target_cash)*100:.1f}%", f"{(actual_bond - target_bond)*100:.1f}%", f"{(actual_stock - target_stock)*100:.1f}%"],
            "Capital (VND)": [f"{capital_cash:,.0f}", f"{capital_bond:,.0f}", f"{capital_stock:,.0f}"]
        })

        st.markdown("### Target vs Actual Allocation Table")
        st.dataframe(df_compare, use_container_width=True, height=250)

        # Cảnh báo nếu lệch hơn 5%
        large_deviation = []
        for i, row in df_compare.iterrows():
            diff_val = abs(float(row['Difference'].strip('%')))
            if diff_val > 5:
                large_deviation.append(row['Asset Class'])

        if large_deviation:
            st.warning(f"⚠️ Allocation deviates significantly (>5%) for: {', '.join(large_deviation)}")
