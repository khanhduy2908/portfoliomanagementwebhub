import matplotlib.pyplot as plt
import streamlit as st

def run(capital_alloc, capital_rf, tickers):
    # Chuẩn bị dữ liệu
    labels = ['Risk-Free Asset'] + tickers
    sizes = [capital_rf] + [capital_alloc[t] for t in tickers]

    if any(s < 0 for s in sizes):
        st.warning("Cannot plot pie chart due to negative allocations.")
        return

    # Tạo biểu đồ với style dark nhẹ
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

    # Căn giữa và định dạng
    for text in texts:
        text.set_color('white')
    for autotext in autotexts:
        autotext.set_color('white')

    ax.set_title("Optimal Complete Portfolio Allocation", fontsize=13, color='white')
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor('#1e1e1e')

    st.pyplot(fig)
