import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

def run(capital_alloc, capital_rf, capital_risky, tickers):
    if not capital_alloc:
        st.error("⚠️ Capital allocation dictionary is empty.")
        return
    if not tickers:
        st.error("⚠️ No tickers provided for visualization.")
        return
    if capital_rf is None or capital_risky is None:
        st.error("⚠️ Risk-free or risky capital is missing.")
        return

    # Build sizes and labels safely
    sizes = []
    labels = []

    if capital_rf > 0:
        sizes.append(capital_rf)
        labels.append("Risk-Free Asset")

    for t in tickers:
        if t in capital_alloc:
            sizes.append(capital_alloc[t])
            labels.append(t)
        else:
            st.warning(f"⚠️ Ticker '{t}' missing in capital_alloc")

    total = sum(sizes)
    if total == 0:
        st.error("⚠️ Total capital is zero. Cannot compute allocation.")
        return

    percentages = [s / total * 100 for s in sizes]

    # Visualization layout
    col1, col2 = st.columns([2, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#1e1e1e')
        colors = plt.cm.Set3.colors[:len(labels)]

        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            textprops={'color': 'white', 'fontsize': 10}
        )

        for text in texts + autotexts:
            text.set_color('white')

        ax.set_title("Complete Portfolio Allocation", fontsize=12, color='white')
        fig.patch.set_facecolor('#1e1e1e')
        ax.set_facecolor('#1e1e1e')
        st.pyplot(fig)

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
