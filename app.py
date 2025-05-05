import streamlit as st
import pandas as pd
import datetime
import config

from utils import (
    block_a_data,
    block_b_factor,
    block_c_covariance,
    block_d_forecast,
    block_e_feasibility,
    block_f_backtest,
    block_g_optimization,
    block_h_complete_portfolio,
    block_i_performance_analysis,
    block_j_stress_testing,
)

# Load danh sách mã cổ phiếu
with open("utils/valid_tickers.txt", "r") as f:
    valid_tickers = sorted([line.strip() for line in f if line.strip()])

# Cấu hình trang
st.set_page_config(page_title="Portfolio Optimization Platform", layout="wide")

st.title("Institutional Portfolio Optimization Platform")

# --- CẤU HÌNH NGƯỜI DÙNG (SIDEBAR) ---
st.sidebar.header("Input Parameters")

tickers_user = st.sidebar.multiselect(
    label="Stock Universe", 
    options=valid_tickers, 
    default=[x for x in ["VNM", "FPT", "MWG", "REE", "VCB"] if x in valid_tickers]
)

if len(tickers_user) < 3:
    st.sidebar.warning("Please select at least 3 tickers.")
    st.stop()

benchmark_user = st.sidebar.selectbox("Benchmark Index", options=valid_tickers, index=valid_tickers.index("VNINDEX") if "VNINDEX" in valid_tickers else 0)

start_user = st.sidebar.date_input("Start Date", value=datetime.date(2020, 1, 1))
end_user = st.sidebar.date_input("End Date", value=datetime.date.today())
rf_user = st.sidebar.number_input("Risk-Free Rate (annual, %)", value=9.0) / 100
capital_user = st.sidebar.number_input("Initial Capital (VND)", value=750_000_000)
A_user = st.sidebar.slider("Risk Aversion Coefficient (A)", min_value=10, max_value=40, value=15)

run_analysis = st.sidebar.button("Run Optimization")

# Cập nhật biến cấu hình toàn cục
config.tickers = tickers_user
config.benchmark_symbol = benchmark_user
config.start_date = pd.to_datetime(start_user)
config.end_date = pd.to_datetime(end_user)
config.rf_annual = rf_user * 100
config.rf = rf_user / 12
config.total_capital = capital_user
config.A = A_user

# --- MAIN EXECUTION PIPELINE ---
if run_analysis:
    st.markdown("### Portfolio Optimization Report")
    st.write("Running full analysis pipeline. Please wait...")

    progress = st.progress(0)

    try:
        # BLOCK A - Data Collection
        data_stocks, data_benchmark, returns_pivot_stocks, returns_benchmark, portfolio_combinations = block_a_data.run(
            config.tickers, config.benchmark_symbol, config.start_date, config.end_date
        )
        progress.progress(10)

        # BLOCK B - Fundamental Filtering
        selected_tickers, selected_combinations, latest_data = block_b_factor.run(data_stocks, returns_benchmark)
        progress.progress(20)

        # BLOCK C - Covariance Estimation
        cov_matrix_dict = block_c_covariance.run(selected_combinations, returns_pivot_stocks)
        progress.progress(30)

        # BLOCK D - Return Forecasting
        adj_returns_combinations, model_store, features_df = block_d_forecast.run(data_stocks, selected_tickers, selected_combinations)
        progress.progress(40)

        # BLOCK E - Feasibility Check
        valid_combinations = block_e_feasibility.run(adj_returns_combinations, cov_matrix_dict)
        progress.progress(50)

        # BLOCK F - Walkforward Backtest
        walkforward_df, error_by_stock = block_f_backtest.run(valid_combinations, features_df)
        progress.progress(60)

        # BLOCK G - Portfolio Optimization
        hrp_cvar_results = block_g_optimization.run(valid_combinations, adj_returns_combinations, cov_matrix_dict, returns_benchmark)
        progress.progress(70)

        # BLOCK H - Complete Portfolio Construction
        best_portfolio, y_capped, capital_alloc, sigma_c, expected_rc, weights, tickers_portfolio = block_h_complete_portfolio.run(
            hrp_cvar_results, adj_returns_combinations, cov_matrix_dict,
            config.rf, config.A, config.total_capital
        )
        progress.progress(80)

        # OUTPUT DISPLAY: Multi-tab Layout
        tabs = st.tabs([
            "1. Data Summary",
            "2. Stock Selection",
            "3. Risk Estimation",
            "4. Return Forecast",
            "5. Feasibility",
            "6. Backtest",
            "7. Optimization",
            "8. Complete Portfolio",
            "9. Performance Evaluation",
            "10. Stress Testing"
        ])

        with tabs[0]:
            st.subheader("1. Market Data Overview")
            st.dataframe(data_stocks.head(10))
            st.line_chart(returns_benchmark)

        with tabs[1]:
            st.subheader("2. Selected Stocks and Portfolio Universe")
            st.write("Selected tickers:", selected_tickers)
            st.dataframe(latest_data[selected_tickers])

        with tabs[2]:
            st.subheader("3. Covariance Matrix Sample")
            sample_key = list(cov_matrix_dict.keys())[0]
            st.write(f"Sample portfolio: {sample_key}")
            st.dataframe(pd.DataFrame(cov_matrix_dict[sample_key]))

        with tabs[3]:
            st.subheader("4. Forecasted Returns")
            st.dataframe(adj_returns_combinations.head())

        with tabs[4]:
            st.subheader("5. Valid Portfolios")
            st.write(f"Total feasible portfolios: {len(valid_combinations)}")

        with tabs[5]:
            st.subheader("6. Rolling Backtest Results")
            st.dataframe(walkforward_df.tail(10))
            st.line_chart(walkforward_df)

        with tabs[6]:
            st.subheader("7. Optimized Risk Portfolio (HRP + CVaR)")
            st.dataframe(hrp_cvar_results.head())

        with tabs[7]:
            st.subheader("8. Final Capital Allocation (Complete Portfolio)")
            st.write("Portfolio weights:")
            st.write(weights)
            st.write("Capital allocation (VND):")
            st.write(capital_alloc)

        with tabs[8]:
            st.subheader("9. Comprehensive Performance Evaluation")
            block_i_performance_analysis.run(
                best_portfolio, returns_pivot_stocks, returns_benchmark,
                config.rf, config.A, config.total_capital,
                data_stocks, data_benchmark, config.benchmark_symbol,
                weights, tickers_portfolio, config.start_date, config.end_date
            )

        with tabs[9]:
            st.subheader("10. Stress Test Results")
            block_j_stress_testing.run(
                best_portfolio, latest_data, data_stocks, returns_pivot_stocks, config.rf
            )

        st.success("Portfolio optimization completed successfully.")
        progress.progress(100)

    except Exception as e:
        st.error(f"Pipeline execution failed: {str(e)}")
