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
    block_g1_bond_model_advanced,
    block_h_complete_portfolio,
    block_h1_visualization,
    block_h2_visualization,
    block_h3_visualization,
    block_i_performance_analysis,
    block_i1_visualization,
    block_i2_visualization,
    block_j_stress_testing
)

# --- Helper functions ---
def map_risk_score_to_A(score):
    if 10 <= score <= 17:
        return 25 - (score - 10) * (10 / 7)
    elif 18 <= score <= 27:
        return 15 - (score - 18) * (10 / 9)
    elif 28 <= score <= 40:
        return 5 - (score - 28) * (4 / 12)
    else:
        raise ValueError("Risk score must be between 10 and 40.")

def get_risk_profile_description(score):
    if 10 <= score <= 17:
        return "Very Conservative – Capital Preservation Focus"
    elif 18 <= score <= 27:
        return "Moderate – Balanced Growth and Preservation"
    elif 28 <= score <= 40:
        return "Aggressive – Growth Focused"
    else:
        return "Undefined"

# Load valid tickers
with open("utils/valid_tickers.txt", "r") as f:
    valid_tickers = sorted([line.strip() for line in f if line.strip()])

# Page config
st.set_page_config(page_title="Portfolio Optimization Platform", layout="wide")
st.title("Institutional-Grade Portfolio Optimization Suite")
st.markdown("---")

# ==============================
# Section 1: Investment Universe
# ==============================
st.sidebar.header("1. Investment Universe")
tickers_user = st.sidebar.multiselect(
    "Select Stocks for Portfolio Construction",
    options=valid_tickers,
    default=["HAH", "TCB", "DGC", "MWG", "POW"]
)

benchmark_user = st.sidebar.selectbox(
    "Benchmark Index",
    options=valid_tickers,
    index=valid_tickers.index("VNINDEX") if "VNINDEX" in valid_tickers else 0
)

start_user = st.sidebar.date_input("Backtest Start Date", datetime.date(2020, 1, 1))
end_user = st.sidebar.date_input("Backtest End Date", datetime.date.today())

capital_user = st.sidebar.number_input("Total Capital (VND)", value=1_000_000_000)
rf_user = st.sidebar.number_input("Annual Risk-Free Rate (%)", value=9.0) / 100

# ================================
# Section 2: Risk Profile & Horizon
# ================================
st.sidebar.header("2. Risk Profile and Investment Horizon")

risk_score_user = st.sidebar.slider("Risk Tolerance Score (10: Conservative, 40: Aggressive)", 10, 40, 25)

def map_risk_score_to_A(score):
    if 10 <= score <= 17:
        return 25 - (score - 10) * (10 / 7)
    elif 18 <= score <= 27:
        return 15 - (score - 18) * (10 / 9)
    elif 28 <= score <= 40:
        return 5 - (score - 28) * (4 / 12)
    else:
        return 15

A_user = map_risk_score_to_A(risk_score_user)

if risk_score_user <= 17:
    risk_level = "Lower"
elif risk_score_user <= 27:
    risk_level = "Moderate"
else:
    risk_level = "Higher"

time_horizon_input = st.sidebar.selectbox("Investment Horizon", ["3–5 years", "6–10 years", "11+ years"])

# Auto strategy mapping
allocation_matrix = {
    ("Lower", "3–5 years"): {"cash": 1.00, "bond": 0.00, "stock": 0.00, "strategy": "All Cash"},
    ("Lower", "6–10 years"): {"cash": 0.30, "bond": 0.50, "stock": 0.20, "strategy": "Strategy 1"},
    ("Lower", "11+ years"): {"cash": 0.10, "bond": 0.30, "stock": 0.60, "strategy": "Strategy 3"},
    ("Moderate", "3–5 years"): {"cash": 0.30, "bond": 0.50, "stock": 0.20, "strategy": "Strategy 1"},
    ("Moderate", "6–10 years"): {"cash": 0.20, "bond": 0.40, "stock": 0.40, "strategy": "Strategy 2"},
    ("Moderate", "11+ years"): {"cash": 0.00, "bond": 0.20, "stock": 0.80, "strategy": "Strategy 4"},
    ("Higher", "3–5 years"): {"cash": 0.20, "bond": 0.40, "stock": 0.40, "strategy": "Strategy 2"},
    ("Higher", "6–10 years"): {"cash": 0.10, "bond": 0.30, "stock": 0.60, "strategy": "Strategy 3"},
    ("Higher", "11+ years"): {"cash": 0.00, "bond": 0.00, "stock": 1.00, "strategy": "Strategy 5"},
}
allocation = allocation_matrix.get((risk_level, time_horizon_input), {
    "cash": 0.2, "bond": 0.4, "stock": 0.4, "strategy": "Default"
})

base_cash = allocation['cash']
base_bond = allocation['bond']
base_stock = allocation['stock']
alpha = max(0, min(1, (25 - A_user) / 23))

alloc_stock = base_stock * (1 - 0.4 * alpha)
alloc_cash = base_cash + base_stock * 0.4 * alpha * 0.5
alloc_bond = 1.0 - alloc_cash - alloc_stock

st.sidebar.markdown(f"**Mapped Strategy:** {allocation['strategy']}")
st.sidebar.markdown(f"**Target Allocation**")
st.sidebar.markdown(f"- Cash: {allocation['cash']*100:.0f}%")
st.sidebar.markdown(f"- Bonds: {allocation['bond']*100:.0f}%")
st.sidebar.markdown(f"- Stocks: {allocation['stock']*100:.0f}%")
st.sidebar.markdown(f"**Risk Aversion Coefficient (A):** {A_user:.2f}")

# ================================
# Section 3: Bond Investment Input
# ================================
st.sidebar.header("3. Bond Investment (Optional)")
bond_price = st.sidebar.number_input("Bond Market Price", value=1_000_000)
bond_coupon = st.sidebar.number_input("Bond Coupon Rate (%)", value=8.0) / 100
bond_face = st.sidebar.number_input("Bond Face Value", value=1_000_000)
bond_years = st.sidebar.number_input("Years to Maturity", value=5)

# ================================
# Section 4: Factor Selection Method
# ================================
st.sidebar.header("4. Factor Selection Strategy")
strategy_options = {
    "Top 2 from each cluster": "top5_by_cluster",
    "Top 5 overall": "top5_overall",
    "Top 5 from strongest clusters": "strongest_clusters"
}
selection_strategy = st.sidebar.selectbox("Select Factor Ranking Method", list(strategy_options.keys()))

# ================================
# Assign to config
# ================================
config.tickers = tickers_user
config.benchmark_symbol = benchmark_user
config.start_date = pd.to_datetime(start_user)
config.end_date = pd.to_datetime(end_user)
config.rf_annual = rf_user * 100
config.rf = rf_user / 12
config.total_capital = capital_user
config.A = A_user
config.risk_score = risk_score_user
config.time_horizon = time_horizon_input
config.strategy_code = allocation['strategy']
config.alloc_cash = alloc_cash
config.alloc_bond = alloc_bond
config.alloc_stock = alloc_stock
config.factor_selection_strategy = strategy_options[selection_strategy]

# Run Trigger
run_analysis = st.sidebar.button("Run Portfolio Optimization")

# --- Validation ---
if not config.tickers or config.benchmark_symbol is None:
    st.error("Please select at least one stock ticker and a benchmark.")
    st.stop()

# --- Execution Pipeline ---
if run_analysis:
    st.subheader("Execution Results")
    with st.spinner("Running institutional-grade optimization pipeline..."):
        try:
            # A. Load and prepare price data
            data_stocks, data_benchmark, returns_pivot_stocks, returns_benchmark, portfolio_combinations = block_a_data.run(
                config.tickers, config.benchmark_symbol, config.start_date, config.end_date
            )
            st.success("Step A – Data successfully loaded.")

            # B. Factor screening and ranking
            selected_tickers, selected_combinations, latest_data, ranking_df = block_b_factor.run(data_stocks, returns_benchmark)
            st.success("Step B – Factor-based stock screening completed.")

            # C. Covariance estimation (GARCH + Ledoit-Wolf)
            cov_matrix_dict = block_c_covariance.run(selected_combinations, returns_pivot_stocks)
            st.success("Step C – Covariance matrix estimated.")

            # D. Forecast next-period returns using ensemble model
            adj_returns_combinations, model_store, features_df = block_d_forecast.run(
                data_stocks, selected_tickers, selected_combinations
            )
            st.success("Step D – Forward return forecasts generated.")

            # E. Filter feasible portfolios for optimization
            valid_combinations = block_e_feasibility.run(adj_returns_combinations, cov_matrix_dict)
            st.success("Step E – Portfolio feasibility screening completed.")

            # F. Backtest model stability using walk-forward validation
            factor_cols = ['Return_Close', 'Return_Volume', 'Spread_HL', 'Volatility_Close', 'Ticker_Encoded']
            walkforward_df, error_by_stock = block_f_backtest.run(valid_combinations, features_df, factor_cols)
            st.success("Step F – Walk-forward backtesting completed.")

            # G. Optimize portfolio using HRP and soft-CVaR constraints
            hrp_result_dict, results_ef = block_g_optimization.run(
                valid_combinations, adj_returns_combinations, cov_matrix_dict, returns_benchmark
            )
            st.success("Step G – HRP + CVaR portfolio optimization executed.")

            # G1. Evaluate custom bond from user input
            bond_return, bond_volatility, bond_label = block_g1_bond_model_advanced.run(
                bond_price=bond_price,
                coupon_rate=bond_coupon,
                face_value=bond_face,
                years_to_maturity=bond_years
            )
            st.success("Step G1 – Bond model parameters calculated.")

            # H. Construct complete portfolio considering cash/bond/stock allocation strategy
            best_portfolio, y_capped, capital_alloc, sigma_c, expected_rc, weights, tickers_portfolio, portfolio_info, sigma_p, mu, y_opt, mu_p, cov = block_h_complete_portfolio.run(
                hrp_result_dict, adj_returns_combinations, cov_matrix_dict,
                rf=config.rf, A=config.A, total_capital=config.total_capital, risk_score=config.risk_score,
                alloc_cash=config.alloc_cash, alloc_bond=config.alloc_bond, alloc_stock=config.alloc_stock,
                y_min=config.y_min, y_max=config.y_max
            )
            st.success("Step H – Final portfolio allocation with integrated strategy computed.")

            # H1. Summary of portfolio and expected utility
            block_h1_visualization.display_portfolio_info(portfolio_info, allocation_matrix, risk_level, time_horizon_input)

            # H2. Capital allocation visualization
            st.session_state["target_stock_ratio"] = config.alloc_stock
            block_h2_visualization.run(capital_alloc=capital_alloc, capital_cash=portfolio_info['capital_cash'], capital_bond=portfolio_info['capital_bond'], capital_stock=portfolio_info['capital_risky'], tickers=tickers_portfolio)
            
            # H3. Efficient Frontier with Capital Allocation Line
            benchmark_return_mean = returns_benchmark['Benchmark_Return'].mean()
            block_h3_visualization.run(
                best_portfolio=hrp_result_dict,
                mu_p=mu_p, sigma_p=sigma_p, rf=config.rf,
                sigma_c=sigma_c, expected_rc=expected_rc,
                y_capped=y_capped, y_opt=y_opt,
                adj_returns_combinations=adj_returns_combinations,
                cov_matrix_dict=cov_matrix_dict,
                simulate_for_visual=True
            )

            # I. Analyze overall performance (risk/return, drawdown, CVaR, alpha/beta)
            block_i_performance_analysis.run(
                best_portfolio, returns_pivot_stocks, returns_benchmark,
                config.rf, config.A, config.total_capital,
                data_stocks, data_benchmark, config.benchmark_symbol,
                weights, tickers_portfolio, config.start_date, config.end_date
            )

            # I1, I2. Visualization of monthly returns, benchmark comparison
            block_i1_visualization.run(returns_pivot_stocks, tickers_portfolio, config.rf, config.start_date, config.end_date)
            block_i2_visualization.run(data_stocks, data_benchmark, config.benchmark_symbol, weights, tickers_portfolio, config.start_date, config.end_date, config.rf)

            # J. Stress testing against macro shocks, sectoral events, historical drawdowns
            block_j_stress_testing.run(best_portfolio, latest_data, data_stocks, returns_pivot_stocks, config.rf)
            st.success("Step J – Multi-layer stress test completed.")

            st.success("All steps completed successfully.")

        except Exception as e:
            st.error(f"Pipeline execution failed: {str(e)}")
