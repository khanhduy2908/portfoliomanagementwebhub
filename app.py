import streamlit as st
import pandas as pd
import numpy as np

from utils import (
    data_loader, factor_ranking, return_forecast, covariance_estimation,
    portfolio_optimizer, complete_allocation, performance_eval, stress_test
)
import config

# --- Page Setup ---
st.set_page_config(page_title="📊 Portfolio Optimizer Pro", layout="wide")

st.title("📈 Portfolio Optimization Dashboard")

# --- Sidebar Configuration ---
st.sidebar.title("⚙️ Portfolio Settings")

tickers = st.sidebar.multiselect("Select Stock Tickers", options=["VNM", "FPT", "MWG", "VCB", "REE"], default=["VNM", "FPT", "MWG", "VCB", "REE"])
benchmark_symbol = st.sidebar.text_input("Benchmark Symbol", value="VNINDEX")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))

rf_annual = st.sidebar.number_input("Annual Risk-Free Rate (%)", value=9.0) / 100
rf = rf_annual / 12
total_capital = st.sidebar.number_input("Total Capital (VND)", value=750_000_000)
A = st.sidebar.slider("Risk Aversion Coefficient (A)", min_value=1, max_value=10, value=5)

run_analysis = st.sidebar.button("🚀 Run Portfolio Optimization")

# --- Run Pipeline ---
if run_analysis:

    st.markdown("## 🔄 Running Portfolio Optimization Pipeline...")

    # --- Summary config ---
    st.info("📌 Configuration:")
    st.json({
        "Tickers": tickers,
        "Benchmark": benchmark_symbol,
        "Start Date": str(start_date),
        "End Date": str(end_date),
        "Risk-Free Rate (monthly)": rf,
        "Total Capital": total_capital,
        "Risk Aversion": A
    })

    try:
        # --- BLOCK A ---
        with st.spinner("📥 Loading and preprocessing data..."):
            data_stocks, data_benchmark, returns_pivot_stocks, returns_benchmark = data_loader.load_data(
                tickers=tickers,
                benchmark_symbol=benchmark_symbol,
                start_date=start_date,
                end_date=end_date
            )
            st.success("✅ Data loaded successfully")

        # --- BLOCK B ---
        with st.spinner("📊 Ranking stocks by factors..."):
            selected_tickers, selected_combinations, latest_data = factor_ranking.rank_stocks(
                data_stocks, returns_benchmark
            )
            st.success(f"✅ Top selected tickers: {selected_tickers}")

        # --- BLOCK C ---
        with st.spinner("📐 Estimating covariance matrices..."):
            cov_matrix_dict = covariance_estimation.batch_covariance_estimation(
                selected_combinations, returns_pivot_stocks
            )
            st.success("✅ Covariance matrices estimated")

        # --- BLOCK D ---
        with st.spinner("🤖 Forecasting returns..."):
            adj_returns_combinations, model_store, features_df = return_forecast.forecast_returns(
                selected_combinations, selected_tickers, data_stocks
            )
            st.success("✅ Return forecasts complete")

        # --- BLOCK E ---
        with st.spinner("🧪 Prechecking feasible portfolios..."):
            valid_combinations = portfolio_optimizer.precheck_portfolios(
                adj_returns_combinations, cov_matrix_dict
            )
            st.success(f"✅ Valid combinations found: {len(valid_combinations)}")

        # --- BLOCK F ---
        with st.spinner("🔁 Evaluating walkforward performance..."):
            walkforward_df, best_combo, best_weights, error_by_stock = return_forecast.walkforward_evaluation(
                valid_combinations, features_df
            )

        # --- BLOCK G ---
        with st.spinner("📊 Running portfolio optimization..."):
            hrp_cvar_results = portfolio_optimizer.optimize_portfolios(
                valid_combinations, adj_returns_combinations, cov_matrix_dict, returns_benchmark
            )
            st.success("✅ Optimization completed")

        # --- BLOCK H ---
        with st.spinner("💼 Constructing complete portfolio..."):
            capital_alloc, best_portfolio, y_capped = complete_allocation.construct_complete_portfolio(
                hrp_cvar_results, adj_returns_combinations, cov_matrix_dict,
                rf, A, total_capital
            )
            st.success("✅ Portfolio constructed")

        # --- BLOCK I ---
        with st.spinner("📈 Evaluating portfolio performance..."):
            fig_perf, summary_df, benchmark_df = performance_eval.evaluate_performance(
                best_portfolio, returns_pivot_stocks, returns_benchmark, rf, A, total_capital
            )
            st.pyplot(fig_perf)
            st.dataframe(summary_df.round(4), use_container_width=True)

        # --- BLOCK J ---
        with st.spinner("🔥 Running stress tests..."):
            fig_stress, summary_stress = stress_test.run_stress_test(
                best_portfolio, data_stocks, latest_data, rf
            )
            st.pyplot(fig_stress)
            st.dataframe(summary_stress.round(2), use_container_width=True)

        st.balloons()
        st.success("🎉 Portfolio optimization completed successfully!")

    except Exception as e:
        st.error(f"❌ Error during execution: {e}")
