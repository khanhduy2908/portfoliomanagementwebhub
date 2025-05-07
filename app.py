import streamlit as st
import pandas as pd
import numpy as np
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
    block_h1_visualization,
    block_h2_visualization,
    block_h3_visualization,
    block_i_performance_analysis,
    block_i1_visualization,
    block_i2_visualization,
    block_j_stress_testing
)

# --- App Configuration ---
st.set_page_config(page_title="Institutional Portfolio Optimization", layout="wide")
st.title("Institutional Portfolio Optimization Platform")

# --- Sidebar Inputs ---
st.sidebar.header("User Configuration")

tickers_user = st.sidebar.multiselect("Select Stock Tickers", config.valid_tickers, default=config.default_tickers)
benchmark_symbol = st.sidebar.selectbox("Benchmark Symbol", [config.benchmark_symbol])
risk_score = st.sidebar.slider("Risk Tolerance Score (10 = Low Risk, 40 = High Risk)", 10, 40, config.risk_score)
start_date = st.sidebar.date_input("Start Date", config.start_date)
end_date = st.sidebar.date_input("End Date", datetime.date.today())
capital = st.sidebar.number_input("Total Capital (VND)", value=config.total_capital, step=10_000_000)

run_button = st.sidebar.button("Run Portfolio Optimization")

if run_button:
    with st.spinner("Running Pipeline..."):

        # Map risk score to A
        A = config.map_risk_score_to_A(risk_score)
        rf = config.rf
        rf_annual = config.rf_annual
        y_min = config.y_min
        y_max = config.y_max

        # --- Block A: Load Data ---
        data_stocks, data_benchmark, returns_pivot_stocks, returns_benchmark, portfolio_labels = \
            block_a_data.run(tickers_user, benchmark_symbol, start_date, end_date)

        # --- Block B: Factor Construction & Selection ---
        selected_tickers, selected_combinations, latest_data, ranking_df = \
            block_b_factor.run(data_stocks, returns_benchmark)

        # --- Block C: Covariance Estimation ---
        cov_matrix_dict = block_c_covariance.run(selected_combinations, returns_pivot_stocks)

        # --- Block D: Return Forecasting (Stacked Model) ---
        adj_returns_combinations, model_store, features_df = \
            block_d_forecast.run(data_stocks, selected_tickers, selected_combinations)

        # --- Block E: Portfolio Feasibility Check ---
        valid_combinations = block_e_feasibility.run(adj_returns_combinations, cov_matrix_dict)

        # --- Block F: Forecast Model Evaluation ---
        walkforward_df, error_by_stock = block_f_backtest.run(
            valid_combinations, features_df, config.forecast_factors,
            lookback=config.lookback, min_samples=config.min_samples
        )

        # --- Block G: Optimization (HRP + CVaR) ---
        hrp_result_dict, results_ef = block_g_optimization.run(
            valid_combinations, adj_returns_combinations, cov_matrix_dict,
            returns_benchmark, config.alpha_cvar, config.lambda_cvar,
            config.beta_l2, config.cvar_soft_limit, config.n_simulations
        )

        # --- Block H: Construct Complete Portfolio ---
        best_portfolio, y_capped, capital_alloc, sigma_c, expected_rc, weights, tickers_portfolio, \
            portfolio_info, sigma_p, mu, y_opt, mu_p, cov = block_h_complete_portfolio.run(
                hrp_result_dict, adj_returns_combinations, cov_matrix_dict,
                rf, A, capital, risk_score, y_min, y_max
        )

        # --- Block H1: Display Portfolio Info ---
        alloc_df = pd.DataFrame({"Ticker": tickers_portfolio, "Weight": weights})
        block_h1_visualization.display_portfolio_info(portfolio_info, alloc_df)

        # --- Block H2: Pie Chart Capital Allocation ---
        block_h2_visualization.run(capital_alloc, portfolio_info['capital_rf'], portfolio_info['capital_risky'], tickers_portfolio)

        # --- Block H3: Efficient Frontier & CAL ---
        block_h3_visualization.run(
            hrp_result_dict, returns_benchmark['Benchmark_Return'].mean(), results_ef,
            best_portfolio, mu_p, sigma_p, rf, sigma_c, expected_rc, y_capped, y_opt, tickers_portfolio, cov
        )

        # --- Block I: Portfolio Performance Analysis ---
        summary_df, regression_df = block_i_performance_analysis.run(
            best_portfolio, returns_pivot_stocks, returns_benchmark,
            rf, A, capital, data_stocks, data_benchmark, benchmark_symbol,
            weights, tickers_portfolio, start_date, end_date
        )

        # --- Block I1: Asset Risk/Return Visualization ---
        block_i1_visualization.run(returns_pivot_stocks, tickers_portfolio, rf, start_date, end_date)

        # --- Block I2: Portfolio vs Benchmark Chart ---
        block_i2_visualization.run(data_stocks, data_benchmark, benchmark_symbol,
                                   weights, tickers_portfolio, start_date, end_date, rf)

        # --- Block J: Stress Testing ---
        block_j_stress_testing.run(best_portfolio, latest_data, data_stocks, returns_pivot_stocks, rf)

    st.success("Pipeline completed successfully.")
