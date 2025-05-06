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

# --- Load valid tickers ---
with open("utils/valid_tickers.txt", "r") as f:
    valid_tickers = sorted([line.strip() for line in f if line.strip()])

# --- Streamlit Page Config ---
st.set_page_config(page_title="Institutional Portfolio Optimization", layout="wide")
st.title("Institutional Portfolio Optimization Platform")

# --- Sidebar Inputs ---
st.sidebar.header("User Configuration")

default_tickers = [x for x in ["VNM", "FPT", "MWG", "REE", "VCB"] if x in valid_tickers]
tickers_user = st.sidebar.multiselect("Select stock tickers", options=valid_tickers, default=default_tickers)

default_benchmark = "VNINDEX" if "VNINDEX" in valid_tickers else valid_tickers[0]
benchmark_user = st.sidebar.selectbox("Select benchmark index", options=valid_tickers, index=valid_tickers.index(default_benchmark))

start_user = st.sidebar.date_input("Start date", value=datetime.date(2020, 1, 1))
end_user = st.sidebar.date_input("End date", value=datetime.date.today())
rf_user = st.sidebar.number_input("Annual risk-free rate (%)", value=9.0) / 100
capital_user = st.sidebar.number_input("Total capital (VND)", value=750_000_000)
A_user = st.sidebar.slider("Risk aversion coefficient (A)", min_value=10, max_value=40, value=15)
strategy_options = {
    "Top 2 from each cluster": "top5_by_cluster",
    "Top 5 overall": "top5_overall",
    "Top 5 from strongest clusters": "strongest_clusters"
}
selection_strategy = st.sidebar.selectbox("Factor selection strategy", list(strategy_options.keys()))

run_analysis = st.sidebar.button("🚀 Run Portfolio Optimization")

# --- Assign config values ---
config.tickers = tickers_user
config.benchmark_symbol = benchmark_user
config.start_date = pd.to_datetime(start_user)
config.end_date = pd.to_datetime(end_user)
config.rf_annual = rf_user * 100
config.rf = rf_user / 12
config.total_capital = capital_user
config.A = A_user
config.factor_selection_strategy = strategy_options[selection_strategy]

# --- Input validation ---
if not config.tickers or config.benchmark_symbol is None:
    st.error("❌ Please select at least one stock ticker and one benchmark.")
    st.stop()

# --- Main Execution Pipeline ---
if run_analysis:
    with st.spinner("Running portfolio optimization pipeline..."):
        try:
            data_stocks, data_benchmark, returns_pivot_stocks, returns_benchmark, portfolio_combinations = block_a_data.run(
                config.tickers, config.benchmark_symbol, config.start_date, config.end_date
            )
            st.success("Block A – Data loaded and monthly returns calculated.")

            selected_tickers, selected_combinations, latest_data, ranking_df = block_b_factor.run(data_stocks, returns_benchmark)
            st.success("Block B – Stock ranking using factor analysis completed.")

            cov_matrix_dict = block_c_covariance.run(selected_combinations, returns_pivot_stocks)
            st.success("Block C – Covariance matrix estimation completed.")

            adj_returns_combinations, model_store, features_df = block_d_forecast.run(data_stocks, selected_tickers, selected_combinations)
            st.success("Block D – Return forecasting with ML ensemble completed.")

            valid_combinations = block_e_feasibility.run(adj_returns_combinations, cov_matrix_dict)
            st.success("Block E – Feasible portfolio combinations selected.")

            factor_cols = ['Return_Close', 'Return_Volume', 'Spread_HL', 'Volatility_Close', 'Ticker_Encoded']
            walkforward_df, error_by_stock = block_f_backtest.run(valid_combinations, features_df, factor_cols)
            st.success("Block F – Forecast model backtesting completed.")

            hrp_result_dict, results_ef = block_g_optimization.run(
                valid_combinations, adj_returns_combinations, cov_matrix_dict, returns_benchmark
            )
            st.success("Block G – HRP + CVaR portfolio optimization completed.")

            best_portfolio, y_capped, capital_alloc, sigma_c, expected_rc, weights, tickers_portfolio, portfolio_info, simulated_returns, cov, mu, y_opt = block_h_complete_portfolio.run(
                hrp_result_dict, adj_returns_combinations, cov_matrix_dict,
                config.rf, config.A, config.total_capital
            )
            st.success("Block H – Complete portfolio construction finished.")

            # Prepare DataFrame for allocation
            alloc_df = pd.DataFrame({
                "Ticker": list(capital_alloc.keys()),
                "Allocated Capital (VND)": list(capital_alloc.values())
            })
            block_h1_visualization.display_portfolio_info(portfolio_info, alloc_df)
            st.success("Block H1 – Portfolio summary and capital allocation displayed.")


            block_h2_visualization.run(capital_alloc, config.total_capital, tickers_portfolio)
            st.success("Block H2 – Allocation pie chart displayed.")

            block_h3_visualization.run(
                hrp_result_dict=hrp_result_dict,
                benchmark_return_mean=returns_benchmark['Benchmark_Return'].mean(),
                results_ef=results_ef,
                best_portfolio=best_portfolio,
                mu_p=mu.mean(),
                sigma_p=np.std(simulated_returns @ weights),
                rf=config.rf,
                sigma_c=sigma_c,
                expected_rc=expected_rc,
                y_capped=y_capped,
                y_opt=y_opt,
                tickers=tickers_portfolio
            )
            st.success("Block H3 – Efficient Frontier and CAL displayed.")

            block_i_performance_analysis.run(
                best_portfolio, returns_pivot_stocks, returns_benchmark,
                config.rf, config.A, config.total_capital,
                data_stocks, data_benchmark, config.benchmark_symbol,
                weights, tickers_portfolio, config.start_date, config.end_date
            )
            st.success("Block I – Portfolio performance evaluation done.")

            block_i1_visualization.run(
                returns_pivot_stocks, tickers_portfolio, config.rf,
                config.start_date, config.end_date
            )
            block_i2_visualization.run(
                data_stocks, data_benchmark, config.benchmark_symbol,
                weights, tickers_portfolio,
                config.start_date, config.end_date, config.rf
            )

            block_j_stress_testing.run(
                best_portfolio, latest_data, data_stocks, returns_pivot_stocks, config.rf
            )
            st.success("Block J – Multi-layer portfolio stress testing completed.")

            st.success("✅ Portfolio optimization pipeline completed successfully.")

        except Exception as e:
            st.error(f"❌ Pipeline execution failed: {str(e)}")
