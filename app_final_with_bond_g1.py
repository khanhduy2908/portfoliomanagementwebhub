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

# --- Load tickers ---
with open("utils/valid_tickers.txt", "r") as f:
    valid_tickers = sorted([line.strip() for line in f if line.strip()])

# --- UI ---
st.set_page_config(page_title="Portfolio Optimization Platform", layout="wide")
st.title("Portfolio Optimization Platform")
st.sidebar.header("Configuration")

tickers_user = st.sidebar.multiselect("Stock Tickers", options=valid_tickers, default=["VNM", "FPT", "MWG", "REE", "VCB"])
benchmark_user = st.sidebar.selectbox("Benchmark Index", options=valid_tickers, index=valid_tickers.index("VNINDEX") if "VNINDEX" in valid_tickers else 0)
start_user = st.sidebar.date_input("Start Date", value=datetime.date(2020, 1, 1))
end_user = st.sidebar.date_input("End Date", value=datetime.date.today())
rf_user = st.sidebar.number_input("Annual Risk-Free Rate (%)", value=9.0) / 100
capital_user = st.sidebar.number_input("Total Capital (VND)", value=750_000_000)

# --- Risk & Time Horizon ---
risk_score_user = st.sidebar.slider("Risk Tolerance Score", min_value=10, max_value=40, value=25)
A_user = map_risk_score_to_A(risk_score_user)
st.sidebar.markdown(f"**Risk Profile**: {get_risk_profile_description(risk_score_user)}")
st.sidebar.markdown(f"**Mapped Risk Aversion (A)**: {A_user:.2f}")

time_horizon_input = st.sidebar.selectbox("Investment Time Horizon", ["3–5 years", "6–10 years", "11+ years"])
if risk_score_user <= 17:
    risk_level = "Lower"
elif risk_score_user <= 27:
    risk_level = "Moderate"
else:
    risk_level = "Higher"

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

st.sidebar.markdown(f"**Mapped Risk Level:** {risk_level}")
st.sidebar.markdown(f"**Selected Strategy:** {allocation['strategy']}")
st.sidebar.markdown(f"**Target Allocation:**\n- Cash: {allocation['cash']*100:.0f}%\n- Bonds: {allocation['bond']*100:.0f}%\n- Stocks: {allocation['stock']*100:.0f}%")

# --- G1: Bond Info Input (before pipeline) ---
st.sidebar.subheader("Bond Information (Optional)")
bond_price = st.sidebar.number_input("Bond Market Price (VND)", value=1000000)
bond_coupon = st.sidebar.number_input("Coupon Rate (%)", value=8.0) / 100
bond_face = st.sidebar.number_input("Bond Face Value (VND)", value=1000000)
bond_years = st.sidebar.number_input("Years to Maturity", value=5)

# --- Assign to config ---
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
config.alloc_cash = allocation['cash']
config.alloc_bond = allocation['bond']
config.alloc_stock = allocation['stock']

# Run button
run_analysis = st.sidebar.button("Run Portfolio Optimization")