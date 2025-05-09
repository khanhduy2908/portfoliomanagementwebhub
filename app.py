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
    block_i_performance_analysis,
    block_i1_visualization,
    block_i2_visualization,
    block_j_stress_testing
)

# --- Helper functions ---
def map_risk_score_to_A(score):
    if 10 <= score <= 17:
        return 30
    elif 18 <= score <= 27:
        return 15
    elif 28 <= score <= 40:
        return 5
    else:
        raise ValueError("Risk score must be between 10 and 40.")

def get_risk_profile_description(score):
    if 10 <= score <= 17:
        return "Low – Capital preservation focus"
    elif 18 <= score <= 27:
        return "Medium – Balanced growth and preservation"
    elif 28 <= score <= 40:
        return "High – Growth focus"
    else:
        return "Undefined"

# --- Load valid tickers ---
with open("utils/valid_tickers.txt", "r") as f:
    valid_tickers = sorted([line.strip() for line in f if line.strip()])

# --- Page config ---
st.set_page_config(page_title="Portfolio Optimization Platform", layout="wide")
st.title("Portfolio Optimization Platform")
st.sidebar.header("Configuration")

# --- Sidebar inputs ---
default_tickers = [x for x in ["VNM", "FPT", "MWG", "REE", "VCB"] if x in valid_tickers]
tickers_user = st.sidebar.multiselect("Stock Tickers", options=valid_tickers, default=default_tickers)

default_benchmark = "VNINDEX" if "VNINDEX" in valid_tickers else valid_tickers[0]
benchmark_user = st.sidebar.selectbox("Benchmark Index", options=valid_tickers, index=valid_tickers.index(default_benchmark))

start_user = st.sidebar.date_input("Start Date", value=datetime.date(2020, 1, 1))
end_user = st.sidebar.date_input("End Date", value=datetime.date.today())
rf_user = st.sidebar.number_input("Annual Risk-Free Rate (%)", value=9.0) / 100
capital_user = st.sidebar.number_input("Total Capital (VND)", value=750_000_000)

risk_score_user = st.sidebar.slider("Risk Tolerance Score", min_value=10, max_value=40, value=25)
st.sidebar.markdown(f"**Risk Profile**: {get_risk_profile_description(risk_score_user)}")

strategy_options = {
    "Top 2 from each cluster": "top5_by_cluster",
    "Top 5 overall": "top5_overall",
    "Top 5 from strongest clusters": "strongest_clusters"
}
selection_strategy = st.sidebar.selectbox("Factor Selection Strategy", list(strategy_options.keys()))
run_analysis = st.sidebar.button("Run Portfolio Optimization")

# --- Assign to config ---
config.tickers = tickers_user
config.benchmark_symbol = benchmark_user
config.start_date = pd.to_datetime(start_user)
config.end_date = pd.to_datetime(end_user)
config.rf_annual = rf_user * 100
config.rf = rf_user / 12
config.total_capital = capital_user
config.A = map_risk_score_to_A(risk_score_user)
config.risk_score = risk_score_user
config.factor_selection_strategy = strategy_options[selection_strategy]
config.y_min = 0.6
config.y_max = 0.9

# --- Input Validation ---
if not config.tickers or config.benchmark_symbol is None:
    st.error("Please select at least one stock ticker and a benchmark.")
    st.stop()

# --- Pipeline ---
if run_analysis:
    with st.spinner("Executing portfolio optimization pipeline..."):
        try:
            data_stocks, data_benchmark, returns_pivot_stocks, returns_benchmark, portfolio_combinations = block_a_data.run(
                config.tickers, config.benchmark_symbol, config.start_date, config.end_date
            )
            st.success("A – Data Loaded")

            selected_tickers, selected_combinations, latest_data, ranking_df = block_b_factor.run(data_stocks, returns_benchmark)
            st.success("B – Factor Ranking Complete")

            cov_matrix_dict = block_c_covariance.run(selected_combinations, returns_pivot_stocks)
            st.success("C – Covariance Matrix Calculated")

            adj_returns_combinations, model_store, features_df = block_d_forecast.run(data_stocks, selected_tickers, selected_combinations)
            st.success("D – Forecasting Complete")

            valid_combinations = block_e_feasibility.run(adj_returns_combinations, cov_matrix_dict)
            st.success("E – Feasible Portfolios Identified")

            factor_cols = ['Return_Close', 'Return_Volume', 'Spread_HL', 'Volatility_Close', 'Ticker_Encoded']
            walkforward_df, error_by_stock = block_f_backtest.run(valid_combinations, features_df, factor_cols)
            st.success("F – Model Backtested")

            hrp_result_dict, results_ef = block_g_optimization.run(
                valid_combinations, adj_returns_combinations, cov_matrix_dict, returns_benchmark
            )
            st.success("G – HRP + CVaR Optimization Done")

            best_portfolio, y_capped, capital_alloc, sigma_c, expected_rc, weights, tickers_portfolio, portfolio_info, sigma_p, mu, y_opt, mu_p, cov = block_h_complete_portfolio.run(
                hrp_result_dict, adj_returns_combinations, cov_matrix_dict,
                config.rf, config.A, config.total_capital, config.risk_score,
                y_min=config.y_min, y_max=config.y_max
            )
            st.success("H – Optimal Portfolio Constructed")

            alloc_df = pd.DataFrame({
                "Ticker": list(capital_alloc.keys())
            })
            block_h1_visualization.display_portfolio_info(portfolio_info)
            st.success("H1 – Portfolio Summary Displayed")

            block_h2_visualization.run(capital_alloc, portfolio_info['capital_rf'], portfolio_info['capital_risky'], tickers_portfolio)
            st.success("H2 – Allocation Visualized")

            block_i_performance_analysis.run(
                best_portfolio, returns_pivot_stocks, returns_benchmark,
                config.rf, config.A, config.total_capital,
                data_stocks, data_benchmark, config.benchmark_symbol,
                weights, tickers_portfolio, config.start_date, config.end_date
            )
            st.success("I – Performance Analyzed")

            block_i1_visualization.run(returns_pivot_stocks, tickers_portfolio, config.rf, config.start_date, config.end_date)
            block_i2_visualization.run(data_stocks, data_benchmark, config.benchmark_symbol, weights, tickers_portfolio, config.start_date, config.end_date, config.rf)

            block_j_stress_testing.run(best_portfolio, latest_data, data_stocks, returns_pivot_stocks, config.rf)
            st.success("J – Stress Testing Completed")

            st.success("✅ Pipeline successfully completed.")

        except Exception as e:
            st.error(f"❌ Pipeline execution failed: {str(e)}")
