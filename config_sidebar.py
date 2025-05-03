# config_sidebar.py
import streamlit as st
import pandas as pd
import os
import sys

# Add current directory to sys.path to resolve local imports
sys.path.append(os.path.dirname(__file__))

from config import (
    benchmark_symbol as default_benchmark,
    start_date as default_start_date,
    rf_annual as default_rf_annual,
    total_capital as default_total_capital,
    A as default_A
)

def sidebar_config():
    st.sidebar.title("Configuration")

    tickers = st.sidebar.text_input("Stock Tickers (comma-separated)", value="VNM,FPT,MWG,VCB,REE")
    tickers = [x.strip().upper() for x in tickers.split(",") if x.strip()]

    benchmark_symbol = st.sidebar.text_input("Benchmark Symbol", value=default_benchmark)
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime(default_start_date))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))

    rf_annual = st.sidebar.number_input("Annual Risk-Free Rate (%)", value=default_rf_annual * 100.0) / 100.0
    rf = rf_annual / 12

    total_capital = st.sidebar.number_input("Total Capital (VND)", value=default_total_capital)
    A = st.sidebar.slider("Risk Aversion Coefficient (A)", min_value=1, max_value=10, value=default_A)

    run_analysis = st.sidebar.button("🚀 Run Portfolio Optimization")

    return tickers, benchmark_symbol, start_date, end_date, rf_annual, rf, total_capital, A, run_analysis
