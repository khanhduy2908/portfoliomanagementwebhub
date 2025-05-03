import streamlit as st
import pandas as pd

from portfolio_app.config import (
    DEFAULT_TICKERS,
    DEFAULT_BENCHMARK,
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
    DEFAULT_RF_ANNUAL,
    DEFAULT_TOTAL_CAPITAL,
    DEFAULT_RISK_AVERSION
)

def sidebar_config():
    st.sidebar.header("📌 Configuration")

    tickers = st.sidebar.text_input("Stock Tickers (comma-separated)", value=','.join(DEFAULT_TICKERS))
    tickers = [x.strip().upper() for x in tickers.split(",") if x.strip()]

    benchmark_symbol = st.sidebar.text_input("Benchmark Symbol", value=DEFAULT_BENCHMARK)

    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime(DEFAULT_START_DATE))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime(DEFAULT_END_DATE))

    rf_annual = st.sidebar.number_input("Annual Risk-Free Rate (%)", value=DEFAULT_RF_ANNUAL) / 100
    rf = rf_annual / 12

    total_capital = st.sidebar.number_input("Total Capital (VND)", value=DEFAULT_TOTAL_CAPITAL)
    A = st.sidebar.slider("Risk Aversion Coefficient (A)", min_value=1, max_value=10, value=DEFAULT_RISK_AVERSION)

    run_analysis = st.sidebar.button("🚀 Run Portfolio Optimization")

    return tickers, benchmark_symbol, start_date, end_date, rf_annual, rf, total_capital, A, run_analysis
