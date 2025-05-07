import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

def run(capital_alloc, capital_rf, capital_risky, tickers):
    if not capital_alloc or not tickers or capital_rf is None or capital_risky is None:
        st.error("⚠️ Missing capital allocation inputs.")
        return

    try:
        # Tạo danh sách sizes cho việc phân bổ vốn (bao gồm cả tài sản không có rủi ro)
        sizes = [capital_rf]  # Khởi tạo với capital_rf (tài sản không rủi ro)
        
        # Kiểm tra và lấy giá trị từ capital_alloc cho mỗi ticker
        for t in tickers:
            sizes.append(capital_alloc.get(t, 0))  # Nếu ticker không có trong capital_alloc, gán giá trị mặc định là 0
    except KeyError as e:
        st.error(f"⚠️ Missing allocation for ticker: {e}")
        return

    # Các nhãn cho biểu đồ pie chart
    labels = ['Risk-Free Asset'] + tickers
    total = capital_rf + capital_risky  # Tổng vốn đầu tư
    if total == 0:
        st.error("⚠️ Total capital is zero. Cannot compute allocation.")
        return

    # Tính tỷ lệ phần trăm cho từng tài sản
    percentages = [s / total * 100 for s in sizes]

    # Chia giao diện thành 2 cột
    col1, col2 = st.columns([2, 1])

    with col1:
        # Vẽ Pie Chart để phân bổ vốn
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#1e1e1e')
        colors = plt.cm.Set3.colors[:len(labels)]

        # Tạo biểu đồ Pie
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            textprops={'color': 'white', 'fontsize': 10}
        )

        # Định dạng màu sắc cho text trong Pie chart
        for text in texts:
            text.set_color('white')
        for autotext in autotexts:
            autotext.set_color('white')

        ax.set_title("Capital Allocation: Risk-Free vs Risky Assets", fontsize=12, color='white')
        fig.patch.set_facecolor('#1e1e1e')
        ax.set_facecolor('#1e1e1e')
        st.pyplot(fig)

    with col2:
        # Tạo bảng phân bổ vốn
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

        # Hiển thị bảng phân bổ vốn
        st.markdown("**Capital Breakdown**")
        st.dataframe(summary_df, use_container_width=True, height=260)
