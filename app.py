
import streamlit as st
import pandas as pd
import datetime
import config
from utils import block_a_data
from utils import block_b_factor
from utils import block_c_covariance
from utils import block_d_forecast
from utils import block_e_feasibility
from utils import block_e1_visualization
from utils import block_e2_visualization
from utils import block_f_backtest
from utils import block_g_optimization
from utils import block_h_complete_portfolio
from utils import block_i_performance_analysis
from utils import block_j_stress_testing

# --- App Config ---
st.set_page_config(page_title="Institutional Portfolio Optimization", layout="wide")
st.title("Institutional Portfolio Optimization Platform")

# --- Sidebar: User Inputs ---
st.sidebar.header("Configuration")

# Load valid tickers
with open("utils/valid_tickers.txt", "r") as f:
    valid_tickers = sorted([line.strip() for line in f if line.strip()])

# Default user config
default_tickers = ["VNM", "FPT", "MWG"]
tickers = st.sidebar.multiselect("Select Stock Tickers", valid_tickers, default=default_tickers)
benchmark_symbol = st.sidebar.selectbox("Select Benchmark", ["VNINDEX", "VN30", "HNXINDEX"])
start_date = st.sidebar.date_input("Start Date", datetime.date(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())
rf_annual = st.sidebar.number_input("Annual Risk-Free Rate (%)", value=9.0)
total_capital = st.sidebar.number_input("Total Capital (VND)", value=750_000_000)
risk_aversion = st.sidebar.slider("Risk Aversion (A)", 10, 40, value=15)

# Derived config
rf_monthly = rf_annual / 12 / 100
A = risk_aversion
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# --- Run Pipeline Button ---
if st.sidebar.button("Run Portfolio Optimization"):
    with st.spinner("Loading and processing data..."):

        # --- Block A ---
        data_stocks, data_benchmark, returns_pivot_stocks, returns_benchmark, portfolio_combinations =             block_a_data.run(tickers, benchmark_symbol, start_date, end_date)

        # --- Block B ---
        adj_returns_combinations = block_b_factor.run(portfolio_combinations, returns_pivot_stocks)

        # --- Block C ---
        cov_matrix_dict = block_c_covariance.run(portfolio_combinations, returns_pivot_stocks)

        # --- Block D ---
        forecasted_returns_dict = block_d_forecast.run(portfolio_combinations, returns_pivot_stocks)

        # --- Block E ---
        valid_combinations = block_e_feasibility.run(adj_returns_combinations, cov_matrix_dict)

        # --- Block E1 Visualization ---
        block_e1_visualization.run(returns_pivot_stocks, tickers, rf_monthly, start_date, end_date)

        # --- Block E2 Visualization ---
        block_e2_visualization.run(data_stocks, data_benchmark, benchmark_symbol, tickers, rf_monthly, start_date, end_date)

        # --- Block F ---
        backtest_results = block_f_backtest.run(returns_pivot_stocks, valid_combinations, adj_returns_combinations)

        # --- Block G ---
        hrp_cvar_results = block_g_optimization.run(valid_combinations, adj_returns_combinations, cov_matrix_dict, rf_monthly)

        # --- Block H ---
        weights, tickers_portfolio = block_h_complete_portfolio.run(hrp_cvar_results, adj_returns_combinations, cov_matrix_dict, rf_monthly, A)

        # --- Block I ---
        block_i_performance_analysis.run(data_stocks, data_benchmark, weights, tickers_portfolio, benchmark_symbol, start_date, end_date, rf_monthly, A, adj_returns_combinations, cov_matrix_dict)

        # --- Block J ---
        block_j_stress_testing.run(tickers_portfolio, returns_pivot_stocks, weights)

    st.success("Portfolio optimization completed successfully.")
