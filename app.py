import streamlit as st
import pandas as pd
import datetime
import config

# --- Import blocks ---
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
    block_e1_visualization,
    block_e2_visualization,
    block_j_stress_testing
)

# --- Load valid tickers ---
with open("utils/valid_tickers.txt", "r") as f:
    valid_tickers = sorted([line.strip() for line in f if line.strip()])

# --- Streamlit UI Configuration ---
st.set_page_config(page_title="Institutional Portfolio Optimization", layout="wide")
st.title("Institutional Portfolio Optimization Platform")

# --- Sidebar Inputs ---
st.sidebar.header("User Configuration")

# Ticker selection
default_tickers = [x for x in ["VNM", "FPT", "MWG", "REE", "VCB"] if x in valid_tickers]
tickers_user = st.sidebar.multiselect("Chọn mã cổ phiếu", options=valid_tickers, default=default_tickers)

# Benchmark selection
default_benchmark = "VNINDEX" if "VNINDEX" in valid_tickers else valid_tickers[0]
benchmark_user = st.sidebar.selectbox("Chọn chỉ số benchmark", options=valid_tickers, index=valid_tickers.index(default_benchmark))

# Date & Parameter Config
start_user = st.sidebar.date_input("Ngày bắt đầu", value=datetime.date(2020, 1, 1))
end_user = st.sidebar.date_input("Ngày kết thúc", value=datetime.date.today())
rf_user = st.sidebar.number_input("Lãi suất phi rủi ro hàng năm (%)", value=9.0) / 100
capital_user = st.sidebar.number_input("Tổng số vốn (VND)", value=750_000_000)
A_user = st.sidebar.slider("Hệ số ngại rủi ro (A)", min_value=10, max_value=40, value=15)

# Run button
run_analysis = st.sidebar.button("🚀 Bắt đầu tối ưu danh mục")

# --- Assign to config ---
config.tickers = tickers_user
config.benchmark_symbol = benchmark_user
config.start_date = pd.to_datetime(start_user)
config.end_date = pd.to_datetime(end_user)
config.rf_annual = rf_user * 100
config.rf = rf_user / 12
config.total_capital = capital_user
config.A = A_user

# --- Check input validity ---
if not config.tickers or config.benchmark_symbol is None:
    st.error("❌ Bạn phải chọn ít nhất 1 mã cổ phiếu và một mã benchmark.")
    st.stop()
    
# --- Main Execution Flow ---
if run_analysis:
    with st.spinner("Running portfolio optimization..."):
        try:
            # A
            data_stocks, data_benchmark, returns_pivot_stocks, returns_benchmark, portfolio_combinations = block_a_data.run(
                config.tickers, config.benchmark_symbol, config.start_date, config.end_date
            )
            st.success("Block A – Data loading and monthly return calculation completed.")

            # B
            selected_tickers, selected_combinations, latest_data, ranking_df = block_b_factor.run(data_stocks, returns_benchmark)
            st.success("Block B – Factor-based stock ranking completed.")

            # C
            cov_matrix_dict = block_c_covariance.run(selected_combinations, returns_pivot_stocks)
            st.success("Block C – Covariance matrix estimation completed.")

            # D
            adj_returns_combinations, model_store, features_df = block_d_forecast.run(data_stocks, selected_tickers, selected_combinations)
            st.success("Block D – Return forecasting using machine learning completed.")

            # E
            valid_combinations = block_e_feasibility.run(adj_returns_combinations, cov_matrix_dict)
            st.success("Block E – Portfolio feasibility check completed.")

            # F
            factor_cols = ['Return_Close', 'Return_Volume', 'Spread_HL', 'Volatility_Close', 'Ticker_Encoded']
            walkforward_df, error_by_stock = block_f_backtest.run(valid_combinations, features_df, factor_cols)
            st.success("Block F – Forecast model backtesting completed.")

            # G
            hrp_cvar_results = block_g_optimization.run(valid_combinations, adj_returns_combinations, cov_matrix_dict, returns_benchmark)
            st.success("Block G – Portfolio optimization with HRP and CVaR completed.")

            # H
            best_portfolio, y_capped, capital_alloc, sigma_c, expected_rc, weights, tickers_portfolio = block_h_complete_portfolio.run(
                hrp_cvar_results, adj_returns_combinations, cov_matrix_dict,
                config.rf, config.A, config.total_capital
            )
            st.success("Block H – Complete portfolio construction completed.")

            # I
            block_i_performance_analysis.run(
                best_portfolio, returns_pivot_stocks, returns_benchmark,
                config.rf, config.A, config.total_capital,
                data_stocks, data_benchmark, config.benchmark_symbol,
                weights, tickers_portfolio, config.start_date, config.end_date
            )
            st.success("Block I – Portfolio performance evaluation completed.")

            # E1
            block_e1_visualization.run(
                returns_pivot_stocks, tickers_portfolio, config.rf,
                config.start_date, config.end_date
            )
            st.success("Block I1 – Historical performance visualization completed.")

            # E2
            block_e2_visualization.run(
                data_stocks, data_benchmark, config.benchmark_symbol,
                weights, tickers_portfolio,
                config.start_date, config.end_date, config.rf
            )
            st.success("Block I2 – Benchmark comparison visualization completed.")

            # J
            block_j_stress_testing.run(
                best_portfolio, latest_data, data_stocks, returns_pivot_stocks, config.rf
            )
            st.success("Block J – Multi-layer stress testing completed.")

            st.success("All portfolio optimization blocks completed successfully.")

        except Exception as e:
            st.error(f"Pipeline execution failed: {str(e)}")
