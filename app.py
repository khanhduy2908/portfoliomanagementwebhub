import streamlit as st
import pandas as pd
import warnings

# Tắt warning
warnings.filterwarnings("ignore")

# --- Sidebar Inputs ---
st.sidebar.header("Portfolio Optimization Inputs")
tickers = st.sidebar.text_input("Stock Tickers (comma-separated)", value="VNM,FPT,MWG,VCB,REE")
tickers = [x.strip().upper() for x in tickers.split(",") if x.strip()]
benchmark_symbol = st.sidebar.text_input("Benchmark Symbol", value="VNINDEX")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))
rf_annual = st.sidebar.number_input("Annual Risk-Free Rate (%)", value=9.0) / 100
rf = rf_annual / 12
total_capital = st.sidebar.number_input("Total Capital (VND)", value=750_000_000)
A = st.sidebar.slider("Risk Aversion Coefficient (A)", min_value=1, max_value=10, value=5)

run_analysis = st.sidebar.button("Run Portfolio Optimization")

# --- Start Processing ---
if run_analysis:
    st.title("Institutional-Grade Portfolio Optimization System")

    with st.spinner("Running Data Processing and Optimization..."):

        # BLOCK A
        from utils.block_a_data import run_block_a
        data_stocks, data_benchmark, returns_pivot_stocks, returns_benchmark, portfolio_combinations = run_block_a(
            tickers, benchmark_symbol, start_date, end_date
        )

        # BLOCK B
        from utils.block_b_factor import run_block_b
        selected_tickers, selected_combinations, latest_data = run_block_b(data_stocks, returns_benchmark)

        # BLOCK C
        from utils.block_c_covariance import run_block_c
        cov_matrix_dict = run_block_c(selected_combinations, returns_pivot_stocks)

        # BLOCK D
        from utils.block_d_forecast import run_block_d
        adj_returns_combinations, model_store, features_df = run_block_d(data_stocks, selected_tickers, selected_combinations)

        # BLOCK E
        from utils.block_e_feasibility import run_block_e
        valid_combinations = run_block_e(adj_returns_combinations, cov_matrix_dict)

        # BLOCK F
        from utils.block_f_backtest import run_block_f
        walkforward_df, best_combo = run_block_f(valid_combinations, features_df)

        # BLOCK G
        from utils.block_g_optimization import run_block_g
        hrp_cvar_results, best_portfolio = run_block_g(valid_combinations, cov_matrix_dict, adj_returns_combinations, returns_benchmark)

        # BLOCK H
        from utils.block_h_complete_portfolio import run_block_h
        weights, tickers_portfolio, capital_alloc, y_opt, y_capped, mu_p, sigma_p, expected_rc, sigma_c, U = run_block_h(
            best_portfolio, rf, A, total_capital, adj_returns_combinations, cov_matrix_dict
        )

        # BLOCK I
        from utils.block_i_performance_analysis import run_block_i
        run_block_i(returns_pivot_stocks, returns_benchmark, tickers_portfolio, weights,
                    rf, A, start_date, end_date, data_stocks, data_benchmark, benchmark_symbol)

        # BLOCK J
        from utils.block_j_stress_testing import run_block_j
        run_block_j(tickers_portfolio, weights, data_stocks, latest_data)

    st.success("Portfolio Optimization Completed.")
