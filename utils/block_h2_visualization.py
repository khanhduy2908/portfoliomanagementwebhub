import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

def run(capital_alloc, capital_rf, tickers):
    labels = ['Risk-Free Asset'] + tickers
    sizes = [capital_rf] + [capital_alloc[t] for t in tickers]

    if any(s < 0 for s in sizes):
        st.warning("Cannot display pie chart due to negative capital allocation.")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#1e1e1e')
        colors = plt.cm.Pastel1.colors[:len(labels)]

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
            autotext.set_color('white')

        ax.set_title("Asset Allocation Overview", fontsize=12, color='white')
        fig.patch.set_facecolor('#1e1e1e')
        ax.set_facecolor('#1e1e1e')
        st.pyplot(fig)

    with col2:
        summary_df = pd.DataFrame({
            "Asset": labels,
            "Capital (VND)": [f"{v:,.0f}" for v in sizes]
        })
        st.markdown("**Capital Breakdown**")
        st.dataframe(summary_df, use_container_width=True, height=220)
