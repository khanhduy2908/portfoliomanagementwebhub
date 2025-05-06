import matplotlib.pyplot as plt
import streamlit as st

def run(capital_alloc, capital_rf, tickers):
    labels = ['Risk-Free Asset'] + tickers
    sizes = [capital_rf] + [capital_alloc[t] for t in tickers]

    if any(s < 0 for s in sizes):
        st.warning("⚠️ Cannot display pie chart due to negative capital allocation.")
        return

    # Create pie chart with dark mode style
    fig, ax = plt.subplots(figsize=(7, 6), facecolor='#1e1e1e')
    colors = plt.cm.Pastel1.colors[:len(labels)]

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

    ax.set_title("Optimal Complete Portfolio Allocation", fontsize=13, color='white')
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor('#1e1e1e')

    st.pyplot(fig)
