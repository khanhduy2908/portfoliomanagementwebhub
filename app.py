import streamlit as st
import pandas as pd
import datetime
import config

import utils.block_a_data as block_a_data
import utils.block_b_factor as block_b_factor
import utils.block_c_covariance as block_c_covariance
import utils.block_d_forecast as block_d_forecast
import utils.block_e_feasibility as block_e_feasibility
import utils.block_f_backtest as block_f_backtest
import utils.block_g_optimization as block_g_optimization
import utils.block_h_complete_portfolio as block_h_complete_portfolio
import utils.block_i_performance_analysis as block_i_performance_analysis
import utils.block_e1_visualization as block_e1_visualization
import utils.block_e2_visualization as block_e2_visualization
import utils.block_j_stress_testing as block_j_stress_testing

# --- Load valid tickers ---
with open("utils/valid_tickers.txt", "r") as f:
    valid_tickers = sorted([line.strip() for line in f if line.strip()])

# --- App UI Config ---
st.set_page_config(page_title="Institutional Portfolio Optimization", layout="wide")
st.title("Institutional Portfolio Optimization Platform")

# --- Sidebar: User Configuration ---
st.sidebar.header("User Configuration")

# Default tickers
default_tickers = [x for x in ["VNM", "FPT", "MWG", "REE", "VCB"] if x in valid_tickers]
tickers_user = st.sidebar.multiselect("Select Stock Tickers", options=valid_tickers, default=default_tickers)

# Benchmark
default_benchmark = "VNINDEX" if "VNINDEX" in valid_tickers else valid_tickers[0]
benchmark_user = st.sidebar.selectbox("Select Benchmark", options=valid_tickers, index=valid_tickers.index(default_benchmark))

# Other parameters
start_user = st.sidebar.date_input("Start Date", value=datetime.date(2020, 1, 1))
end_user = st.sidebar.date_input("End Date", value=datetime.date.today())
rf_user = st.sidebar.number_input("Annual Risk-Free Rate (%)", value=9.0) / 100
capital_user = st.sidebar.number_input("Total Capital (VND)", value=750_000_000)
A_user = st.sidebar.slider("Risk Aversion Coefficient (A)", min_value=10, max_value=40, value=15)

# Run button
run_analysis = st.sidebar.button("Run Portfolio Optimization")

# --- Override config ---
config.tickers = tickers_user
config.benchmark_symbol = benchmark_user
config.start_date = pd.to_datetime(start_user)
config.end_date = pd.to_datetime(end_user)
config.rf_annual = rf_user * 100
config.rf = rf_user / 12
config.total_capital = capital_user
config.A = A_user

# --- Main Execution Pipeline ---
if run_analysis:
    with st.spinner("Running portfolio optimization..."):
        try:
            data_stocks, data_benchmark, returns_pivot_stocks, returns_benchmark, portfolio_combinations = block_a_data.run(
                config.tickers, config.benchmark_symbol, config.start_date, config.end_date
            )

            selected_tickers, selected_combinations, latest_data = block_b_factor.run(data_stocks, returns_benchmark)
            cov_matrix_dict = block_c_covariance.run(selected_combinations, returns_pivot_stocks)
            adj_returns_combinations, model_store, features_df = block_d_forecast.run(data_stocks, selected_tickers, selected_combinations)
            valid_combinations = block_e_feasibility.run(adj_returns_combinations, cov_matrix_dict)
            walkforward_df, error_by_stock = block_f_backtest.run(valid_combinations, features_df)
            hrp_cvar_results = block_g_optimization.run(valid_combinations, adj_returns_combinations, cov_matrix_dict, returns_benchmark)

            best_portfolio, y_capped, capital_alloc, sigma_c, expected_rc, weights, tickers_portfolio = block_h_complete_portfolio.run(
                hrp_cvar_results, adj_returns_combinations, cov_matrix_dict,
                config.rf, config.A, config.total_capital
            )

            block_i_performance_analysis.run(
                best_portfolio, returns_pivot_stocks, returns_benchmark,
                config.rf, config.A, config.total_capital,
                data_stocks, data_benchmark, config.benchmark_symbol,
                weights, tickers_portfolio, config.start_date, config.end_date
            )

            block_e1_visualization.run(
                returns_pivot_stocks, tickers_portfolio, config.rf,
                config.start_date, config.end_date
            )

            block_e2_visualization.run(
                data_stocks, data_benchmark, config.benchmark_symbol,
                weights, tickers_portfolio,
                config.start_date, config.end_date, config.rf
            )

            block_j_stress_testing.run(
                best_portfolio, latest_data, data_stocks, returns_pivot_stocks, config.rf
            )

            st.success("Portfolio optimization completed successfully!")

        except Exception as e:
            st.error(f"Pipeline execution failed: {str(e)}")
