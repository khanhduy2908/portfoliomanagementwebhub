import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

def run(capital_alloc, capital_rf, capital_risky, tickers):
    if not capital_alloc or not tickers or capital_rf is None or capital_risky is None:
        st.error("⚠️ Missing capital allocation inputs.")
        return

    # Kiểm tra phân bổ vốn cho các ticker
    sizes = [capital_rf]  # Bắt đầu với capital_rf (tài sản không có rủi ro)
    sizes.extend([capital_alloc.get(t, 0) for t in tickers])  # Lấy phân bổ cho các ticker, mặc định là 0 nếu không có

    labels = ['Risk-Free Asset'] + tickers  # Nhãn cho biểu đồ
    total = capital_rf + capital_risky  # Tổng tài sản

    if total == 0:
        st.error("⚠️ Total capital is zero. Cannot compute allocation.")
        return

    # Tính tỷ lệ phần trăm phân bổ cho mỗi phần tử
    percentages = [s / total * 100 for s in sizes]

    # Sử dụng 2 cột trong Streamlit để chia không gian giữa biểu đồ và bảng phân bổ
    col1, col2 = st.columns([2, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#1e1e1e')
        colors = plt.cm.Set3.colors[:len(labels)]  # Chọn màu sắc từ bảng màu Set3 của matplotlib

        # Vẽ biểu đồ pie
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            textprops={'color': 'white', 'fontsize': 10}
        )

        # Đặt màu cho các văn bản trên biểu đồ
        for text in texts:
            text.set_color('white')
        for autotext in autotexts:
            autotext.set_color('white')

        ax.set_title("Complete Portfolio Allocation", fontsize=12, color='white')
        fig.patch.set_facecolor('#1e1e1e')
        ax.set_facecolor('#1e1e1e')
        st.pyplot(fig)

    with col2:
        # Tạo bảng hiển thị phân bổ vốn
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
