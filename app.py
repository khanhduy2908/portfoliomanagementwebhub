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
    block_e1_visual_asset,
    block_e2_visual_portfolio
)

# --- Page Configuration ---
st.set_page_config(
    page_title="Institutional Portfolio Optimization",
    layout="wide"
)
st.title("Institutional Portfolio Optimization Platform")

# --- Sidebar Configuration ---
st.sidebar.header("User Configuration")

# Load valid stock tickers
with open("utils/valid_tickers.txt", "r") as f:
    valid_tickers = sorted([line.strip() for line in f if line.strip()])

# User Inputs
default_tickers = [x for x in ["VNM", "FPT", "MWG"] if x in valid_tickers]
tickers_user = st.sidebar.multiselect("Select Stock Tickers", options=valid_tickers, default=default_tickers)
benchmark_symbol = st.sidebar.selectbox("Benchmark Index", options=["VNINDEX", "VN30", "VN100", "HNX", "UPCOM"], index=0)
start_date = st.sidebar.date_input("Start Date", value=datetime.date(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime.date.today())
rf_annual = st.sidebar.number_input("Annual Risk-Free Rate (%)", value=9.0, min_value=0.0, max_value=20.0)
total_capital = st.sidebar.number_input("Total Investment Capital (VND)", value=750_000_000, step=10_000_000)
risk_aversion = st.sidebar.slider("Risk Aversion Coefficient (A)", min_value=1, max_value=30, value=15)

# Update config
config.tickers = tickers_user
config.benchmark_symbol = benchmark_symbol
config.start_date = start_date
config.end_date = end_date
config.rf_annual = rf_annual
config.rf = rf_annual / 12 / 100
config.total_capital = total_capital
config.A = risk_aversion

# --- Pipeline Execution ---
if st.button("Run Optimization Pipeline"):
    st.subheader("1. Data Collection")
    data_stocks, data_benchmark, returns_pivot_stocks, returns_benchmark, portfolio_combinations = block_a_data.run()

    st.subheader("2. Factor-Based Ranking")
    selected_tickers, selected_combinations, latest_data = block_b_factor.run(data_stocks, returns_benchmark)

    st.subheader("3. Covariance Matrix Estimation")
    cov_matrix_dict = block_c_covariance.run(selected_combinations, returns_pivot_stocks)

    st.subheader("4. Return Forecasting via Ensemble Models")
    adj_returns_combinations, features_df = block_d_forecast.run(data_stocks, selected_tickers, selected_combinations)

    st.subheader("5. Portfolio Feasibility Check")
    valid_combinations = block_e_feasibility.run(adj_returns_combinations, cov_matrix_dict)

    st.subheader("6. Predictive Backtesting (Walkforward)")
    block_f_backtest.run(valid_combinations, features_df)

    st.subheader("7. Robust Optimization with Tail Risk Control")
    hrp_cvar_results = block_g_optimization.run(valid_combinations, adj_returns_combinations, cov_matrix_dict, returns_benchmark)

    st.subheader("8. Optimal Complete Portfolio Construction")
    best_portfolio = block_h_complete_portfolio.run(hrp_cvar_results, adj_returns_combinations, cov_matrix_dict)

    st.subheader("9. Portfolio Performance Evaluation")
    tickers_portfolio = list(best_portfolio['Weights'].keys())
    weights = list(best_portfolio['Weights'].values())
    block_i_performance_analysis.run(best_portfolio, returns_pivot_stocks, returns_benchmark, tickers_portfolio, weights)

    st.subheader("10. Risk Visualization and Stress Testing")
    block_j_stress_testing.run(best_portfolio, latest_data, data_stocks, returns_pivot_stocks)

    st.subheader("11. Additional Visualization - Asset Risk Bubble")
    block_e1_visual_asset.run(returns_pivot_stocks, tickers_portfolio, config.rf, config.start_date, config.end_date)

    st.subheader("12. Additional Visualization - Efficient Frontier & CAL")
    block_e2_visual_portfolio.run(
        data_stocks=data_stocks,
        data_benchmark=data_benchmark,
        benchmark_symbol=benchmark_symbol,
        weights=np.array(weights),
        tickers_portfolio=tickers_portfolio,
        start_date=start_date,
        end_date=end_date,
        rf_monthly=config.rf,
        best_portfolio=best_portfolio,
        adj_returns_combinations=adj_returns_combinations,
        cov_matrix_dict=cov_matrix_dict,
        A=config.A
    )
