import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import streamlit as st
import pandas as pd

def get_contrast_color(rgb):
    """Chọn màu chữ (đen hoặc trắng) dựa trên độ sáng nền."""
    r, g, b = rgb
    brightness = r * 0.299 + g * 0.587 + b * 0.114
    return 'black' if brightness > 186 else 'white'

def run(capital_alloc, capital_rf, capital_risky, tickers):
    if not capital_alloc or not tickers or capital_rf is None or capital_risky is None:
        st.error("⚠️ Missing capital allocation inputs.")
        return

    sizes = [capital_rf] + [capital_alloc.get(t, 0) for t in tickers]
    labels = ['Risk-Free Asset'] + tickers
    total = capital_rf + capital_risky

    if total == 0:
        st.error("⚠️ Total capital is zero. Cannot compute allocation.")
        return

    percentages = [s / total * 100 for s in sizes]
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
            textprops={'fontsize': 10}
        )

        for i, autotext in enumerate(autotexts):
            rgb = wedges[i].get_facecolor()[:3]
            rgb_scaled = [int(c * 255) for c in rgb]
            color = get_contrast_color(rgb_scaled)
            autotext.set_color(color)
            autotext.set_path_effects([
                path_effects.Stroke(linewidth=1.5, foreground='black'),
                path_effects.Normal()
            ])

        for text in texts:
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
        st.markdown("Capital in VND. Allocation rounded to 0.1%.")
        st.dataframe(summary_df, use_container_width=True, height=260)
