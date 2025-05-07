import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def run(capital_alloc, capital_rf, capital_risky, tickers):
    if not capital_alloc or not tickers or capital_rf is None or capital_risky is None:
        st.error("⚠️ Missing required data to visualize capital allocation.")
        return

    labels = ['Risk-Free Asset'] + tickers
    values = [capital_rf] + [capital_alloc.get(t, 0) for t in tickers]
    total = sum(values)

    if total == 0:
        st.error("⚠️ Total capital is zero. Cannot visualize allocation.")
        return

    percentages = [v / total * 100 for v in values]

    # Bar Chart Visualization
    fig, ax = plt.subplots(figsize=(8, 4), facecolor='#1e1e1e')
    bars = ax.bar(labels, percentages, color=plt.cm.Paired.colors)

    ax.set_facecolor('#1e1e1e')
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_title("Capital Allocation (%)", color='white')
    ax.set_ylabel("Percentage (%)", color='white')
    ax.set_ylim(0, 100)
    ax.tick_params(colors='white')

    for bar, pct in zip(bars, percentages):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{pct:.1f}%", ha='center', va='bottom', color='white', fontsize=8)

    st.pyplot(fig)

    # Summary Table
    df_summary = pd.DataFrame({
        "Asset": labels,
        "Allocated Capital (VND)": [f"{v:,.0f}" for v in values],
        "Allocation (%)": [f"{p:.1f}%" for p in percentages]
    })

    st.markdown("### Capital Allocation Table")
    st.dataframe(df_summary, use_container_width=True)
