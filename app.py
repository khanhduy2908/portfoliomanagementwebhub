import streamlit as st
import pandas as pd
import numpy as np
from vnstock import listing_companies

from utils import (
    data_loader, factor_ranking, return_forecast, covariance_estimation,
    portfolio_optimizer, complete_allocation, performance_eval, stress_test
)
import config

# --- Page Setup ---
st.set_page_config(page_title="📊 Portfolio Optimizer Pro", layout="wide")

# --- Load Vietnamese Stock Tickers ---
try:
    df_listing = listing_companies()
    tickers_all = sorted(df_listing['ticker'].dropna().unique().tolist())
except Exception as e:
    st.error(f"❌ Failed to load tickers from Vnstock: {e}")
    tickers_all = ["VNM", "FPT", "MWG", "VCB", "REE"]

# --- Sidebar Configuration ---
st.sidebar.title("⚙️ Portfolio Configuration")

tickers = st.sidebar.multiselect("Select Stock Tickers", options=tickers_all, default=["VNM", "FPT", "MWG", "VCB", "REE"])
benchmark_symbol = st.sidebar.selectbox("Benchmark Symbol", options=tickers_all, index=tickers_all.index("VNINDEX") if "VNINDEX" in tickers_all else 0)
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))

rf_annual = st.sidebar.number_input("Annual Risk-Free Rate (%)", value=9.0) / 100
rf = rf_annual / 12
total_capital = st.sidebar.number_input("Total Capital (VND)", value=750_000_000)
A = st.sidebar.slider("Risk Aversion Coefficient (A)", min_value=1, max_value=10, value=5)

run_analysis = st.sidebar.button("🚀 Run Portfolio Optimization")

# --- Main App Pipeline ---
if run_analysis:
    st.markdown("## 🔄 Running Portfolio Optimization Pipeline...")

    try:
        # --- Block A ---
        st.markdown("### 📥 Loading Data")
        data_stocks, data_benchmark, returns_pivot_stocks, returns_benchmark = data_loader.load_data(
            tickers=tickers,
            benchmark_symbol=benchmark_symbol,
            start_date=start_date,
            end_date=end_date
        )
        st.success("✅ Data Loaded Successfully")

        # --- Block B ---
        st.markdown("### 🔍 Factor Ranking")
        selected_tickers, selected_combinations, latest_data = factor_ranking.rank_stocks(
            data_stocks, returns_benchmark
        )

        # --- Block C ---
        st.markdown("### 📐 Estimating Covariance Matrices")
        cov_matrix_dict = covariance_estimation.compute_cov_matrices(
            selected_combinations, returns_pivot_stocks
        )

        # --- Block D ---
        st.markdown("### 🤖 Forecasting Returns with ML")
        adj_returns_combinations, model_store, features_df = return_forecast.forecast_returns(
            selected_combinations, selected_tickers, data_stocks
        )

        # --- Block E ---
        st.markdown("### 🧪 Prechecking Feasible Portfolios")
        valid_combinations = portfolio_optimizer.precheck_portfolios(
            adj_returns_combinations, cov_matrix_dict
        )

        # --- Block F ---
        walkforward_df, best_combo, best_weights, error_by_stock = return_forecast.walkforward_evaluation(
            valid_combinations, features_df
        )

        # --- Block G ---
        st.markdown("### 📊 Optimizing Portfolio (HRP + CVaR)")
        hrp_cvar_results = portfolio_optimizer.optimize_portfolios(
            valid_combinations, adj_returns_combinations, cov_matrix_dict, returns_benchmark
        )

        # --- Block H ---
        st.markdown("### 💼 Constructing Final Portfolio (CAL Line)")
        capital_alloc, best_portfolio, y_capped = complete_allocation.construct_complete_portfolio(
            hrp_cvar_results, adj_returns_combinations, cov_matrix_dict,
            rf, A, total_capital
        )

        # --- Block I ---
        st.markdown("### 📈 Performance Evaluation")
        fig_perf, summary_df, benchmark_df = performance_eval.evaluate_performance(
            best_portfolio, returns_pivot_stocks, returns_benchmark, rf, A, total_capital
        )
        st.pyplot(fig_perf)
        st.dataframe(summary_df.round(4), use_container_width=True)

        # --- Block J ---
        st.markdown("### 🔥 Stress Testing (Historical + Monte Carlo)")
        fig_stress, summary_stress = stress_test.run_stress_test(
            best_portfolio, data_stocks, latest_data, rf
        )
        st.pyplot(fig_stress)
        st.dataframe(summary_stress.round(2), use_container_width=True)

        st.success("🎉 Portfolio Optimization Completed Successfully!")

    except Exception as e:
        st.error(f"❌ Error during execution: {e}")
