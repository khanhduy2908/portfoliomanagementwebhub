import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import streamlit as st
import pandas as pd

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

        wedges, texts, _ = ax.pie(
            sizes,
            labels=None,
            startangle=90,
            colors=colors,
            radius=1
        )

        # Add percentage manually with better contrast and style
        for i, p in enumerate(percentages):
            ang = (wedges[i].theta2 + wedges[i].theta1) / 2
            x = 0.7 * np.cos(np.deg2rad(ang))
            y = 0.7 * np.sin(np.deg2rad(ang))
            txt = ax.text(
                x, y, f"{p:.1f}%", ha='center', va='center',
                fontsize=10, color='white', weight='bold'
            )
            txt.set_path_effects([
                path_effects.Stroke(linewidth=1.5, foreground='black'),
                path_effects.Normal()
            ])

        # Add labels
        for i, wedge in enumerate(wedges):
            ang = (wedge.theta2 + wedge.theta1) / 2
            x = 1.15 * np.cos(np.deg2rad(ang))
            y = 1.15 * np.sin(np.deg2rad(ang))
            ax.text(x, y, labels[i], ha='center', va='center', color='white', fontsize=9)

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

    if False:
        st.markdown("### Capital Allocation by Ticker")
        st.dataframe(alloc_df.style.format({"Allocated Capital (VND)": "{:,.0f}"}), use_container_width=True)
