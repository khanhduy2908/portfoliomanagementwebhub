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

# --- Setup ---
st.set_page_config(page_title="Institutional Portfolio Optimization", layout="wide")
st.title("Institutional Portfolio Optimization Platform")

# --- Sidebar Inputs (Dropdown danh sách mã cổ phiếu và benchmark) ---
from vnstock import listing_companies

try:
    all_symbols_df = listing_companies(source='VCI')
    valid_symbols = sorted(all_symbols_df['ticker'].dropna().unique().tolist())
except Exception as e:
    st.sidebar.error(f"Lỗi khi tải danh sách mã cổ phiếu: {e}")
    valid_symbols = ["VNM", "FPT", "MWG", "VCB", "REE", "VNINDEX", "VN30"]

st.sidebar.header("User Configuration")

tickers_user = st.sidebar.multiselect(
    "Chọn mã cổ phiếu",
    options=valid_symbols,
    default=["VNM", "FPT", "MWG", "VCB", "REE"]
)

benchmark_user = st.sidebar.selectbox(
    "Chọn Benchmark",
    options=["VNINDEX", "VN30", "HNXINDEX", "UPCOMINDEX"] + valid_symbols,
    index=0
)

start_user = st.sidebar.date_input("Start Date", value=datetime.date(2020, 1, 1))
end_user = st.sidebar.date_input("End Date", value=datetime.date.today())
rf_user = st.sidebar.number_input("Annual Risk-Free Rate (%)", value=9.0) / 100
capital_user = st.sidebar.number_input("Total Capital (VND)", value=750_000_000)
A_user = st.sidebar.slider("Risk Aversion Coefficient (A)", min_value=1, max_value=10, value=5)

run_analysis = st.sidebar.button("Run Portfolio Optimization")

# --- Overwrite Config ---
config.tickers = tickers_user
config.benchmark_symbol = benchmark_user
config.start_date = pd.to_datetime(start_user)
config.end_date = pd.to_datetime(end_user)
config.rf_annual = rf_user * 100
config.rf = rf_user / 12
config.total_capital = capital_user
config.A = A_user

# --- Run Analysis Pipeline ---
if run_analysis:
    with st.spinner("Running full optimization pipeline..."):
        try:
            with st.container():
                st.subheader("Block A – Data Loading")
                data_stocks, data_benchmark, returns_pivot_stocks, returns_benchmark, portfolio_combinations = block_a_data.run(
                    config.tickers, config.benchmark_symbol, config.start_date, config.end_date
                )
                st.dataframe(returns_pivot_stocks.head())

            with st.container():
                st.subheader("Block B – Factor Filtering")
                selected_tickers, selected_combinations, latest_data = block_b_factor.run(
                    data_stocks, returns_benchmark
                )
                st.write("Selected tickers:", selected_tickers)

            with st.container():
                st.subheader("Block C – Covariance Estimation")
                cov_matrix_dict = block_c_covariance.run(selected_combinations, returns_pivot_stocks)

            with st.container():
                st.subheader("Block D – Return Forecasting")
                adj_returns_combinations, model_store, features_df = block_d_forecast.run(
                    data_stocks, selected_tickers, selected_combinations
                )

            with st.container():
                st.subheader("Block E – Feasibility Check")
                valid_combinations = block_e_feasibility.run(
                    adj_returns_combinations, cov_matrix_dict
                )

            with st.container():
                st.subheader("Block F – Walk-forward Backtest")
                walkforward_df, error_by_stock = block_f_backtest.run(valid_combinations, features_df)
                with st.expander("Backtest Error Summary"):
                    st.dataframe(error_by_stock)

            with st.container():
                st.subheader("Block G – Optimization with CVaR")
                hrp_cvar_results = block_g_optimization.run(
                    valid_combinations, adj_returns_combinations, cov_matrix_dict, returns_benchmark
                )

            with st.container():
                st.subheader("Block H – Complete Portfolio Construction")
                best_portfolio, y_capped, capital_alloc, sigma_c, expected_rc, weights, tickers_portfolio = block_h_complete_portfolio.run(
                    hrp_cvar_results, adj_returns_combinations, cov_matrix_dict,
                    config.rf, config.A, config.total_capital
                )

            with st.container():
                st.subheader("Block I – Performance Analysis & Reporting")
                block_i_performance_analysis.run(
                    best_portfolio, returns_pivot_stocks, returns_benchmark,
                    config.rf, config.A, config.total_capital,
                    data_stocks, data_benchmark, config.benchmark_symbol,
                    weights, tickers_portfolio, config.start_date, config.end_date
                )

            with st.container():
                st.subheader("Block J – Multi-Layer Stress Testing")
                block_j_stress_testing.run(
                    best_portfolio, latest_data, data_stocks, returns_pivot_stocks, config.rf
                )

            st.success("All blocks completed successfully.")
        except Exception as e:
            st.error(f"Pipeline error: {str(e)}")
