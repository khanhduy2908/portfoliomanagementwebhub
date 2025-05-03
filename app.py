# app.py

import streamlit as st
import pandas as pd
import numpy as np
from utils import (
    data_loader, factor_ranking, return_forecast, covariance_estimation,
    portfolio_optimizer, complete_allocation, performance_eval, stress_test
)
from vnstock import listing_companies
import config

# --- Page Setup ---
st.set_page_config(page_title="📊 Portfolio Optimizer Pro", layout="wide")

st.title("📈 Portfolio Optimization Dashboard")
st.markdown("Designed for institutional-grade optimization and risk analysis using ML & CVaR models.")

# --- Sidebar: Smart Configuration ---
st.sidebar.header("⚙️ Portfolio Settings")

@st.cache_data
def get_ticker_metadata():
    try:
        df = listing_companies()
        df = df.dropna(subset=['ticker', 'industry', 'exchange'])
        df = df[df['exchange'].isin(['HOSE', 'HNX', 'UPCOM'])]
        return df
    except Exception as e:
        st.warning(f"⚠️ Failed to load ticker metadata: {e}")
        return pd.DataFrame()

ticker_df = get_ticker_metadata()

# Filter options
industries = sorted(ticker_df['industry'].unique())
exchanges = sorted(ticker_df['exchange'].unique())
selected_exchange = st.sidebar.selectbox("🏛️ Select Exchange", ["All"] + exchanges)
selected_industries = st.sidebar.multiselect("🏭 Filter by Industry", options=industries)

filtered_df = ticker_df.copy()
if selected_exchange != "All":
    filtered_df = filtered_df[filtered_df['exchange'] == selected_exchange]
if selected_industries:
    filtered_df = filtered_df[filtered_df['industry'].isin(selected_industries)]

available_tickers = sorted(filtered_df['ticker'].unique().tolist())

# Ticker + Benchmark Selection
tickers = st.sidebar.multiselect(
    "📌 Select Stock Tickers",
    options=available_tickers,
    default=available_tickers[:5] if len(available_tickers) >= 5 else available_tickers
)

benchmark_map = {
    "HOSE": ["VNINDEX", "VN30"],
    "HNX": ["HNXINDEX"],
    "UPCOM": ["UPCOMINDEX"]
}
benchmark_candidates = benchmark_map.get(selected_exchange, ["VNINDEX", "VN30", "HNXINDEX", "UPCOMINDEX"])
benchmark_symbol = st.sidebar.selectbox("📈 Benchmark Symbol", benchmark_candidates)

# Time & Risk Settings
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))
rf_annual = st.sidebar.number_input("Risk-Free Rate (Annual %)", value=9.0) / 100
rf = rf_annual / 12
total_capital = st.sidebar.number_input("Total Capital (VND)", value=750_000_000)
A = st.sidebar.slider("Risk Aversion Coefficient (A)", 1, 10, value=5)

run_analysis = st.sidebar.button("🚀 Run Portfolio Optimization")

# --- MAIN RUN ---
if run_analysis:
    st.markdown("## 🔄 Running Portfolio Optimization Pipeline...")

    try:
        # --- Block A: Load data ---
        st.markdown("### 📥 Loading Data")
        data_stocks, data_benchmark, returns_pivot_stocks, returns_benchmark, error_logs = data_loader.load_data(
            tickers=tickers,
            benchmark_symbol=benchmark_symbol,
            start_date=start_date,
            end_date=end_date
        )
        st.success("✅ Data Loaded Successfully")

        if error_logs:
            st.warning("⚠️ Missing data for the following tickers:")
            for log in error_logs:
                st.markdown(f"- {log}")

        # --- Block B: Factor Ranking ---
        st.markdown("### 🔍 Factor Ranking")
        selected_tickers, selected_combinations, latest_data = factor_ranking.rank_stocks(
            data_stocks, returns_benchmark
        )

        # --- Block C: Covariance Estimation ---
        st.markdown("### 📐 Estimating Covariance Matrices")
        cov_matrix_dict = covariance_estimation.compute_cov_matrices(
            selected_combinations, returns_pivot_stocks
        )

        # --- Block D: Return Forecast ---
        st.markdown("### 🤖 Forecasting Returns")
        adj_returns_combinations, model_store, features_df = return_forecast.forecast_returns(
            selected_combinations, selected_tickers, data_stocks
        )

        # --- Block E: Precheck ---
        st.markdown("### 🧪 Prechecking Portfolios")
        valid_combinations = portfolio_optimizer.precheck_portfolios(
            adj_returns_combinations, cov_matrix_dict
        )

        # --- Block F: Evaluation ---
        walkforward_df, best_combo, best_weights, error_by_stock = return_forecast.walkforward_evaluation(
            valid_combinations, features_df
        )

        # --- Block G: Portfolio Optimization ---
        st.markdown("### 📊 Optimizing Portfolio (HRP + CVaR)")
        hrp_cvar_results = portfolio_optimizer.optimize_portfolios(
            valid_combinations, adj_returns_combinations, cov_matrix_dict, returns_benchmark
        )

        # --- Block H: Capital Allocation ---
        st.markdown("### 💼 Final Portfolio Allocation")
        capital_alloc, best_portfolio, y_capped = complete_allocation.construct_complete_portfolio(
            hrp_cvar_results, adj_returns_combinations, cov_matrix_dict,
            rf, A, total_capital
        )

        # --- Block I: Performance Evaluation ---
        st.markdown("### 📈 Performance Evaluation")
        fig_perf, summary_df, benchmark_df = performance_eval.evaluate_performance(
            best_portfolio, returns_pivot_stocks, returns_benchmark, rf, A, total_capital
        )
        st.pyplot(fig_perf)
        st.dataframe(summary_df.round(4), use_container_width=True)

        # --- Block J: Stress Test ---
        st.markdown("### 🔥 Stress Testing")
        fig_stress, summary_stress = stress_test.run_stress_test(
            best_portfolio, data_stocks, latest_data, rf
        )
        st.pyplot(fig_stress)
        st.dataframe(summary_stress.round(2), use_container_width=True)

        st.success("🎯 Optimization Completed Successfully!")

    except Exception as e:
        st.error(f"❌ Error during execution: {e}")
