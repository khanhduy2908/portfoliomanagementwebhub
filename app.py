import streamlit as st
import pandas as pd
import numpy as np

from portfolio_app.config import *
from portfolio_app.config_sidebar import sidebar_config

# --- LAYOUT CONFIG ---
st.set_page_config(page_title="Portfolio Optimizer Platform", layout="wide")

# --- SIDEBAR CONFIG ---
tickers, benchmark_symbol, start_date, end_date, rf_annual, rf, total_capital, A, run_analysis = sidebar_config()

# --- RUN PIPELINE IF USER CLICKS ---
if run_analysis:
    with st.spinner("🔄 Running portfolio optimization..."):
        # --- BLOCK A: Data Loading ---
        from portfolio_app.utils import block_a_data
        data_stocks, data_benchmark, returns_pivot_stocks, returns_benchmark, portfolio_combinations = block_a_data.run(tickers, benchmark_symbol, start_date, end_date)

        # --- BLOCK B: Factor Ranking ---
        from portfolio_app.utils import block_b_factor
        selected_tickers, selected_combinations, latest_data = block_b_factor.run(data_stocks, returns_benchmark)

        # --- BLOCK C: Covariance Estimation ---
        from portfolio_app.utils import block_c_covariance
        cov_matrix_dict = block_c_covariance.run(selected_combinations, returns_pivot_stocks)

        # --- BLOCK D: Return Forecasting ---
        from portfolio_app.utils import block_d_forecast
        adj_returns_combinations, features_df, model_store = block_d_forecast.run(selected_combinations, selected_tickers, data_stocks)

        # --- BLOCK E: Feasibility Check ---
        from portfolio_app.utils import block_e_feasibility
        valid_combinations = block_e_feasibility.run(adj_returns_combinations, cov_matrix_dict)

        # --- BLOCK F: Walkforward Backtest ---
        from portfolio_app.utils import block_f_backtest
        walkforward_df, best_combo, error_by_stock = block_f_backtest.run(valid_combinations, features_df)

        # --- BLOCK G: Optimization (HRP + CVaR) ---
        from portfolio_app.utils import block_g_optimization
        hrp_cvar_results, best_portfolio = block_g_optimization.run(valid_combinations, adj_returns_combinations, cov_matrix_dict, returns_benchmark)

        # --- BLOCK H: Complete Portfolio Construction ---
        from portfolio_app.utils import block_h_complete_portfolio
        weights, tickers_portfolio, mu_p, sigma_p, y_opt, y_capped, expected_rc, sigma_c, capital_rf, capital_risky = block_h_complete_portfolio.run(
            best_portfolio, adj_returns_combinations, cov_matrix_dict, total_capital, A, rf
        )

        # --- BLOCK I: Performance Evaluation ---
        from portfolio_app.utils import block_i_performance_analysis
        block_i_performance_analysis.run(
            best_portfolio, returns_pivot_stocks, returns_benchmark, weights, tickers_portfolio,
            start_date, end_date, rf
        )

        # --- BLOCK J: Stress Testing ---
        from portfolio_app.utils import block_j_stress_testing
        block_j_stress_testing.run(
            best_portfolio, data_stocks, tickers_portfolio, latest_data, cov_matrix_dict,
            adj_returns_combinations, returns_benchmark, rf
        )

        st.success("✅ Portfolio optimization completed!")
