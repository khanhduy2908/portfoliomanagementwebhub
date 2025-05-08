import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

def run(capital_alloc: dict, capital_rf: float, capital_risky: float, tickers: list):
    if not capital_alloc or not tickers or capital_rf is None or capital_risky is None:
        st.error("⚠️ Missing capital allocation inputs.")
        return

    # 1. Đồng bộ tickers từ chính capital_alloc keys (tránh mismatch)
    tickers_final = list(capital_alloc.keys())
    sizes = [capital_rf] + [capital_alloc[t] for t in tickers_final]
    labels = ['Risk-Free Asset'] + tickers_final
    total = capital_rf + capital_risky

    if total == 0:
        st.error("⚠️ Total capital is zero. Cannot compute allocation.")
        return

    percentages = [s / total * 100 for s in sizes]

    # 2. Hiển thị hai cột
    col1, col2 = st.columns([2, 1])

    # 3. Biểu đồ Pie
    with col1:
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#1e1e1e')
        cmap = plt.cm.get_cmap('tab20c', len(labels))
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
            autotext.set_color('yellow')

        ax.set_title("Capital Allocation: Risk-Free vs Risky Assets", fontsize=12, color='white')
        ax.set_facecolor('#1e1e1e')
        fig.patch.set_facecolor('#1e1e1e')
        fig.tight_layout()
        st.pyplot(fig)

    # 4. Bảng phân bổ vốn
    with col2:
        summary_df = pd.DataFrame({
            "Asset": labels,
            "Capital (VND)": [f"{v:,.0f}" for v in sizes],
            "Allocation (%)": [f"{p:.1f}%" for p in percentages]
        })
        total_row = pd.DataFrame([{
            "Asset": "Total",
            "Capital (VND)": f"{total:,.0f}",
            "Allocation (%)": "100.0%"
        }])
        summary_df = pd.concat([summary_df, total_row], ignore_index=True)

        st.markdown("**Capital Breakdown**")
        st.dataframe(summary_df, use_container_width=True, height=260)
