import streamlit as st
import pandas as pd
import datetime
import config

# === Load optimization blocks ===
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

# === Load valid stock tickers ===
with open("utils/valid_tickers.txt", "r") as f:
    valid_tickers = sorted([line.strip() for line in f if line.strip()])

# === UI Configuration ===
st.set_page_config(page_title="Institutional Portfolio Optimization", layout="wide")
st.title("Institutional Portfolio Optimization Platform")

# === Sidebar: User Inputs ===
st.sidebar.header("User Configuration")

selected_tickers = st.sidebar.multiselect(
    "Select Stock Tickers",
    options=valid_tickers,
    default=[x for x in ["VNM", "FPT", "MWG", "REE", "VCB"] if x in valid_tickers]
)

benchmark_symbol = st.sidebar.selectbox(
    "Select Benchmark Index",
    options=valid_tickers,
    index=valid_tickers.index("VNINDEX") if "VNINDEX" in valid_tickers else 0
)

start_date = st.sidebar.date_input("Start Date", value=datetime.date(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime.date.today())
rf_annual = st.sidebar.number_input("Annual Risk-Free Rate (%)", value=9.0) / 100
total_capital = st.sidebar.number_input("Total Investment Capital (VND)", value=750_000_000)
risk_aversion = st.sidebar.slider("Risk Aversion Coefficient (A)", min_value=5, max_value=40, value=15)

run_pipeline = st.sidebar.button("Run Portfolio Optimization")

# === Assign to global config ===
config.tickers = selected_tickers
config.benchmark_symbol = benchmark_symbol
config.start_date = pd.to_datetime(start_date)
config.end_date = pd.to_datetime(end_date)
config.rf_annual = rf_annual * 100
config.rf = rf_annual / 12
config.total_capital = total_capital
config.A = risk_aversion

# === Main UI ===
if run_pipeline:
    st.info("Running optimization pipeline. Please wait...")
    progress = st.progress(0)

    try:
        # Block A
        data_stocks, data_benchmark, returns_pivot_stocks, returns_benchmark, portfolio_combinations = block_a_data.run(
            config.tickers, config.benchmark_symbol, config.start_date, config.end_date
        )
        progress.progress(10)

        # Block B
        selected_tickers, selected_combinations, latest_data = block_b_factor.run(data_stocks, returns_benchmark)
        progress.progress(20)

        # Block C
        cov_matrix_dict = block_c_covariance.run(selected_combinations, returns_pivot_stocks)
        progress.progress(30)

        # Block D
        adj_returns_combinations, model_store, features_df = block_d_forecast.run(data_stocks, selected_tickers, selected_combinations)
        progress.progress(40)

        # Block E
        valid_combinations = block_e_feasibility.run(adj_returns_combinations, cov_matrix_dict)
        progress.progress(50)

        # Block F
        walkforward_df, error_by_stock = block_f_backtest.run(valid_combinations, features_df)
        progress.progress(60)

        # Block G
        hrp_cvar_results = block_g_optimization.run(valid_combinations, adj_returns_combinations, cov_matrix_dict, returns_benchmark)
        progress.progress(70)

        # Block H
        best_portfolio, y_capped, capital_alloc, sigma_c, expected_rc, weights, tickers_portfolio = block_h_complete_portfolio.run(
            hrp_cvar_results, adj_returns_combinations, cov_matrix_dict,
            config.rf, config.A, config.total_capital
        )
        progress.progress(80)

        # Tabs for displaying results
        tab_names = [
            "Data Overview", "Factor Analysis", "Return Forecasting", "Portfolio Optimization",
            "Capital Allocation", "Performance Evaluation", "Stress Testing"]
        tabs = st.tabs(tab_names)

        with tabs[0]:
            st.subheader("Market Data Overview")
            st.dataframe(data_stocks.head())
            st.line_chart(returns_benchmark)

        with tabs[1]:
            st.subheader("Top Selected Stocks")
            st.write(selected_tickers)

        with tabs[2]:
            st.subheader("Forecasted Stock Returns")
            st.dataframe(pd.DataFrame(adj_returns_combinations).T.head())

        with tabs[3]:
            st.subheader("HRP + CVaR Optimized Portfolios")
            st.dataframe(pd.DataFrame(hrp_cvar_results).head())

        with tabs[4]:
            st.subheader("Final Capital Allocation")
            st.write(capital_alloc)

        with tabs[5]:
            st.subheader("Portfolio Performance Analysis")
            block_i_performance_analysis.run(
                best_portfolio, returns_pivot_stocks, returns_benchmark,
                config.rf, config.A, config.total_capital,
                data_stocks, data_benchmark, config.benchmark_symbol,
                weights, tickers_portfolio, config.start_date, config.end_date
            )

        with tabs[6]:
            st.subheader("Advanced Stress Testing")
            block_j_stress_testing.run(
                best_portfolio, latest_data, data_stocks, returns_pivot_stocks, config.rf
            )

        st.success("Portfolio optimization completed successfully!")
        progress.progress(100)

    except Exception as e:
        st.error(f"Pipeline execution failed: {str(e)}")
