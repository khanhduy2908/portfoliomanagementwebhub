import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import streamlit as st
import pandas as pd

def run(capital_alloc: dict, capital_cash: float, capital_bond: float, capital_stock: float, tickers: list):
    if capital_alloc is None or capital_stock is None:
        st.error("⚠️ Missing capital allocation inputs.")
        return

    # 1. Đồng bộ tickers để đảm bảo đúng thứ tự
    tickers_final = tickers
    capital_stock_total = sum([capital_alloc[t] for t in tickers_final])

    # Kiểm tra tính nhất quán
    if abs(capital_stock - capital_stock_total) > 1:
        capital_stock = capital_stock_total

    # 2. Chuẩn bị dữ liệu biểu đồ và bảng
    labels = ['Cash', 'Bond'] + tickers_final
    sizes = [capital_cash, capital_bond] + [capital_alloc[t] for t in tickers_final]
    total = sum(sizes)

    if total <= 0:
        st.error("⚠️ Total capital is zero. Cannot compute allocation.")
        return

    percentages = [s / total * 100 for s in sizes]

    col1, col2 = st.columns([2, 1])

    # 3. Biểu đồ Pie
    with col1:
        fig, ax = plt.subplots(figsize=(6, 5), facecolor='#1e1e1e')
        cmap = plt.cm.get_cmap('Set3', len(labels))
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

        ax.set_title("Overall Capital Allocation", fontsize=13, color='white')
        ax.set_facecolor('#1e1e1e')
        fig.patch.set_facecolor('#1e1e1e')
        fig.tight_layout()
        st.pyplot(fig)

    # 4. Bảng phân bổ
    with col2:
        summary_df = pd.DataFrame({
            "Asset Class / Ticker": labels,
            "Capital (VND)": [f"{v:,.0f}" for v in sizes],
            "Allocation (%)": [f"{p:.1f}%" for p in percentages]
        })
        total_row = pd.DataFrame([{
            "Asset Class / Ticker": "Total",
            "Capital (VND)": f"{total:,.0f}",
            "Allocation (%)": "100.0%"
        }])
        summary_df = pd.concat([summary_df, total_row], ignore_index=True)

        st.markdown("### Capital Allocation Table")
        st.dataframe(summary_df, use_container_width=True, height=300)

def run_dynamic_allocation(monthly_alloc_df: pd.DataFrame):
    if monthly_alloc_df is None or 'Month' not in monthly_alloc_df.columns:
        st.warning("No monthly allocation data provided.")
        return

    df_melted = monthly_alloc_df.melt(id_vars='Month', var_name='Asset Class', value_name='Capital (VND)')

    st.markdown("### Projected Asset Allocation Over Time")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.set_style("darkgrid")

    sns.lineplot(data=df_melted, x='Month', y='Capital (VND)', hue='Asset Class', linewidth=2.2, ax=ax)

    ax.set_title("Projected Monthly Allocation: Cash, Bonds, and Stocks", fontsize=14, color='white')
    ax.set_xlabel("Month", fontsize=11, color='white')
    ax.set_ylabel("Capital (VND)", fontsize=11, color='white')
    ax.tick_params(colors='white')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.autofmt_xdate()
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor('#1e1e1e')
    ax.legend(facecolor='#2e2e2e', labelcolor='white')

    st.pyplot(fig)

    st.markdown("### Monthly Allocation Table")
    df_display = monthly_alloc_df.copy()
    df_display['Month'] = df_display['Month'].dt.strftime('%b %Y')
    st.dataframe(df_display, use_container_width=True)
