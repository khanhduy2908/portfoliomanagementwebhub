import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import (
    data_loader, factor_ranking, return_forecast, covariance_estimation,
    portfolio_optimizer, complete_allocation, performance_eval, stress_test
)
import config

# --- PAGE SETUP ---
st.set_page_config(page_title="📊 Portfolio Optimization Dashboard", layout="wide")
st.title("📈 Professional Portfolio Optimization Platform")
st.markdown("""
Welcome to the institutional-grade portfolio optimizer. This tool is designed for asset managers, investment banks,
and financial analysts to build, optimize, and analyze complete portfolios using advanced quantitative techniques.
""")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("⚙️ Configuration")

with st.sidebar.expander("🔽 Input Parameters", expanded=True):
    tickers = st.multiselect("Select Stock Tickers", options=["VNM", "FPT", "MWG", "VCB", "REE"], default=["VNM", "FPT", "MWG", "VCB", "REE"])
    benchmark_symbol = st.text_input("Benchmark Symbol", value="VNINDEX")
    start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("today"))
    rf_annual = st.number_input("Annual Risk-Free Rate (%)", value=9.0)
    rf = rf_annual / 12 / 100
    total_capital = st.number_input("Total Capital (VND)", value=750_000_000)
    A = st.slider("Risk Aversion Coefficient (A)", min_value=1, max_value=10, value=5)

run_analysis = st.sidebar.button("🚀 Run Portfolio Optimization")

# --- RUN MAIN PIPELINE ---
if run_analysis:
    with st.spinner("⏳ Executing full portfolio optimization pipeline..."):

        # --- BLOCK A: Load Data ---
        data_stocks, data_benchmark, returns_pivot_stocks, returns_benchmark = data_loader.load_data(
            tickers=tickers, benchmark_symbol=benchmark_symbol,
            start_date=start_date, end_date=end_date
        )

        # --- BLOCK B: Factor Ranking ---
        selected_tickers, selected_combinations, latest_data = factor_ranking.rank_stocks(
            data_stocks, returns_benchmark
        )

        # --- BLOCK C: Covariance Estimation ---
        cov_matrix_dict = covariance_estimation.compute_cov_matrices(
            selected_combinations, returns_pivot_stocks
        )

        # --- BLOCK D: Return Forecasting ---
        adj_returns_combinations, model_store, features_df = return_forecast.forecast_returns(
            selected_combinations, selected_tickers, data_stocks
        )

        # --- BLOCK E: Portfolio Feasibility ---
        valid_combinations = portfolio_optimizer.precheck_portfolios(
            adj_returns_combinations, cov_matrix_dict
        )

        # --- BLOCK F: Walkforward Evaluation ---
        walkforward_df, best_combo, best_weights, error_by_stock = return_forecast.walkforward_evaluation(
            valid_combinations, features_df
        )

        # --- BLOCK G: Optimization (Robust CVaR + HRP) ---
        hrp_cvar_results = portfolio_optimizer.optimize_portfolios(
            valid_combinations, adj_returns_combinations, cov_matrix_dict, returns_benchmark
        )

        # --- BLOCK H: Capital Allocation Line ---
        capital_alloc, best_portfolio, y_capped = complete_allocation.construct_complete_portfolio(
            hrp_cvar_results, adj_returns_combinations, cov_matrix_dict,
            rf, A, total_capital
        )

        # --- BLOCK I: Performance Evaluation ---
        fig_perf, summary_df, benchmark_df = performance_eval.evaluate_performance(
            best_portfolio, returns_pivot_stocks, returns_benchmark, rf, A, total_capital
        )

        st.markdown("## 📊 Portfolio Performance Summary")
        st.pyplot(fig_perf)
        st.dataframe(summary_df.style.format("{:.4f}"), use_container_width=True)

        # --- BLOCK J: Multi-Layer Stress Testing ---
        fig_stress, summary_stress = stress_test.run_stress_test(
            best_portfolio, data_stocks, latest_data, rf
        )

        st.markdown("## 🔥 Multi-Layer Stress Test Results")
        for fig in fig_stress:
            st.pyplot(fig)
        st.dataframe(summary_stress.style.format("{:.2f}"), use_container_width=True)

    st.success("✅ Portfolio optimization completed successfully!")
