import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

def run(capital_alloc, capital_rf, capital_risky, tickers):
    if not capital_alloc or not tickers or capital_rf is None or capital_risky is None:
        st.error("⚠️ Missing capital allocation inputs.")
        return

    try:
        # Creating the values for capital allocation (including risk-free asset)
        sizes = [capital_rf]
        # Ensure all tickers have a value in capital_alloc
        for t in tickers:
            sizes.append(capital_alloc.get(t, 0))  # Default to 0 if ticker not in capital_alloc
    except KeyError as e:
        st.error(f"⚠️ Missing allocation for ticker: {e}")
        return

    labels = ['Risk-Free Asset'] + tickers
    total = capital_rf + capital_risky
    if total == 0:
        st.error("⚠️ Total capital is zero. Cannot compute allocation.")
        return

    percentages = [s / total * 100 for s in sizes]

    col1, col2 = st.columns([2, 1])

    with col1:
        # Creating Pie Chart for capital allocation
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

        for text in texts:
            text.set_color('white')
        for autotext in autotexts:
            autotext.set_color('white')

        ax.set_title("Capital Allocation: Risk-Free vs Risky Assets", fontsize=12, color='white')
        fig.patch.set_facecolor('#1e1e1e')
        ax.set_facecolor('#1e1e1e')
        st.pyplot(fig)

    with col2:
        # Creating a summary dataframe
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
