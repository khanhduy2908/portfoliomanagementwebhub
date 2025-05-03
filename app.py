import streamlit as st
import pandas as pd
import datetime

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

# --- Sidebar Inputs ---
st.title("Institutional Portfolio Optimization Platform")

st.sidebar.header("User Configuration")
tickers_input = st.sidebar.text_input("Stock Tickers (comma-separated)", value="VNM,FPT,MWG,VCB,REE")
tickers = [x.strip().upper() for x in tickers_input.split(",") if x.strip()]

benchmark_symbol = st.sidebar.text_input("Benchmark Symbol", value="VNINDEX")
start_date = st.sidebar.date_input("Start Date", value=datetime.date(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime.date.today())

rf_annual = st.sidebar.number_input("Annual Risk-Free Rate (%)", value=9.0) / 100
rf = rf_annual / 12
total_capital = st.sidebar.number_input("Total Capital (VND)", value=750_000_000)
A = st.sidebar.slider("Risk Aversion Coefficient (A)", min_value=1, max_value=10, value=5)

run_analysis = st.sidebar.button("Run Portfolio Optimization")

# --- Run Analysis ---
if run_analysis:
    with st.spinner("Running full analysis pipeline..."):
        # BLOCK A
        data_stocks, data_benchmark, returns_pivot_stocks, returns_benchmark, portfolio_combinations = block_a_data.run(
            tickers, benchmark_symbol, start_date, end_date
        )

        # BLOCK B
        selected_tickers, selected_combinations, latest_data = block_b_factor.run(
            data_stocks, returns_benchmark
        )

        # BLOCK C
        cov_matrix_dict = block_c_covariance.run(
            selected_combinations, returns_pivot_stocks
        )

        # BLOCK D
        adj_returns_combinations, model_store, features_df = block_d_forecast.run(
            data_stocks, selected_tickers, selected_combinations
        )

        # BLOCK E
        valid_combinations = block_e_feasibility.run(
            adj_returns_combinations, cov_matrix_dict
        )

        # BLOCK F
        walkforward_df, error_by_stock = block_f_backtest.run(
            valid_combinations, features_df
        )

        # BLOCK G
        hrp_cvar_results = block_g_optimization.run(
            valid_combinations, adj_returns_combinations, cov_matrix_dict, returns_benchmark
        )

        # BLOCK H
        best_portfolio, y_capped, capital_alloc, sigma_c, expected_rc, weights, tickers_portfolio = block_h_complete_portfolio.run(
            hrp_cvar_results, adj_returns_combinations, cov_matrix_dict, rf, A, total_capital
        )

        # BLOCK I
        block_i_performance_analysis.run(
            best_portfolio, returns_pivot_stocks, returns_benchmark,
            rf, A, total_capital, data_stocks, data_benchmark, benchmark_symbol,
            weights, tickers_portfolio, start_date, end_date
        )

        # BLOCK J
        block_j_stress_testing.run(
            best_portfolio, latest_data, data_stocks, returns_pivot_stocks, rf
        )

    st.success("Analysis completed successfully.")
