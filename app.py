# app.py

import streamlit as st
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

# Streamlit Config
st.set_page_config(page_title="Institutional Portfolio Optimization", layout="wide")
st.title("Institutional Portfolio Optimization System")

# --- Sidebar ---
st.sidebar.header("User Configuration")
valid_tickers = ['VNM', 'FPT', 'MWG', 'VCB', 'REE']
default_tickers = ['VNM', 'FPT', 'MWG']

tickers = st.sidebar.multiselect("Select Stock Tickers", valid_tickers, default=default_tickers)
benchmark_symbol = st.sidebar.selectbox("Select Benchmark", ['VNINDEX'])
start_date = st.sidebar.date_input("Start Date", datetime.date(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

rf_annual = st.sidebar.slider("Annual Risk-Free Rate (%)", 0.0, 15.0, 9.0)
rf = rf_annual / 12 / 100

total_capital = st.sidebar.number_input("Total Capital (VND)", value=750_000_000, step=50_000_000)
risk_score = st.sidebar.slider("Risk Tolerance Score", 10, 40, 25)

if 10 <= risk_score <= 17:
    A = 30
elif 18 <= risk_score <= 27:
    A = 15
else:
    A = 5

if st.sidebar.button("Run Portfolio Optimization Pipeline"):

    with st.spinner("Running Block A – Data Loading"):
        config.tickers = tickers
        config.benchmark_symbol = benchmark_symbol
        config.start_date = start_date
        config.end_date = end_date
        config.rf = rf
        config.rf_annual = rf_annual
        config.total_capital = total_capital
        config.risk_score = risk_score
        config.A = A

        data_stocks, data_benchmark, returns_pivot_stocks, returns_benchmark, portfolio_labels = block_a_data.run(
            tickers, benchmark_symbol, start_date, end_date
        )

    with st.spinner("Running Block B – Factor Selection"):
        selected_tickers, selected_combinations, latest_data, ranking_df = block_b_factor.run(
            data_stocks, returns_benchmark
        )

    with st.spinner("Running Block C – Covariance Estimation"):
        cov_matrix_dict = block_c_covariance.run(selected_combinations, returns_pivot_stocks)

    with st.spinner("Running Block D – Return Forecasting"):
        adj_returns_combinations, model_store, features_df = block_d_forecast.run(
            data_stocks, selected_tickers, selected_combinations
        )

    with st.spinner("Running Block E – Feasibility Filtering"):
        valid_combinations = block_e_feasibility.run(adj_returns_combinations, cov_matrix_dict)

    with st.spinner("Running Block F – Forecast Backtesting"):
        factor_cols = ['Return_Close', 'Return_Volume', 'Spread_HL', 'Volatility_Close', 'Ticker_Encoded']
        _ = block_f_backtest.run(valid_combinations, features_df, factor_cols)

    with st.spinner("Running Block G – Optimization with CVaR"):
        hrp_result_dict, results_ef = block_g_optimization.run(
            valid_combinations, adj_returns_combinations, cov_matrix_dict,
            returns_benchmark,
            config.alpha_cvar, config.lambda_cvar, config.beta_l2, config.cvar_soft_limit, config.n_simulations
        )

    with st.spinner("Running Block H – Complete Portfolio Construction"):
        result = block_h_complete_portfolio.run(
            hrp_result_dict, adj_returns_combinations, cov_matrix_dict,
            config.rf, config.A, config.total_capital, config.risk_score,
            config.y_min, config.y_max
        )

        best_portfolio, y_capped, capital_alloc, sigma_c, expected_rc, weights, tickers_portfolio, \
        portfolio_info, sigma_p, mu, y_opt, mu_p, cov = result

    with st.spinner("Block H1 – Portfolio Summary"):
        alloc_df = pd.DataFrame.from_dict(capital_alloc, orient='index', columns=['Capital'])
        alloc_df = alloc_df.sort_values(by='Capital', ascending=False)
        block_h1_visualization.display_portfolio_info(portfolio_info, alloc_df)

    with st.spinner("Block H2 – Capital Allocation Pie Chart"):
        block_h2_visualization.run(capital_alloc, portfolio_info['capital_rf'], portfolio_info['capital_risky'], tickers_portfolio)

    with st.spinner("Block H3 – Efficient Frontier + CAL"):
        block_h3_visualization.run(hrp_result_dict, returns_benchmark['Benchmark_Return'].mean(), results_ef,
                                   best_portfolio, mu_p, sigma_p, config.rf,
                                   sigma_c, expected_rc, y_capped, y_opt, tickers_portfolio, cov)

    with st.spinner("Block I – Portfolio Performance Analysis"):
        block_i_performance_analysis.run(
            best_portfolio, returns_pivot_stocks, returns_benchmark,
            config.rf, config.A, config.total_capital,
            data_stocks, data_benchmark, benchmark_symbol,
            weights, tickers_portfolio, start_date, end_date
        )

    with st.spinner("Block I1 – Asset-Level Analysis"):
        block_i1_visualization.run(returns_pivot_stocks, tickers_portfolio, config.rf, start_date, end_date)

    with st.spinner("Block I2 – Portfolio vs Benchmark"):
        block_i2_visualization.run(data_stocks, data_benchmark, benchmark_symbol,
                                   weights, tickers_portfolio, start_date, end_date, config.rf)

    with st.spinner("Block J – Stress Testing"):
        block_j_stress_testing.run(best_portfolio, latest_data, data_stocks, returns_pivot_stocks, config.rf)
