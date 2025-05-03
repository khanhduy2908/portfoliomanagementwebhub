import streamlit as st
import pandas as pd
import numpy as np

from portfolio_app.config import *

# --- SIDEBAR CONFIG FUNCTION ---
def sidebar_config():
    st.sidebar.title("📊 Portfolio Optimizer Settings")

    tickers = st.sidebar.text_input("Stock Tickers (comma-separated)", value="VNM,FPT,MWG,VCB,REE")
    tickers = [x.strip().upper() for x in tickers.split(",") if x.strip()]

    benchmark_symbol = st.sidebar.text_input("Benchmark Symbol", value=benchmark_symbol)

    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime(start_date))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))

    rf_annual = st.sidebar.number_input("Annual Risk-Free Rate (%)", value=rf_annual * 100) / 100
    rf = rf_annual / 12

    total_capital = st.sidebar.number_input("Total Capital (VND)", value=total_capital)

    A = st.sidebar.slider("Risk Aversion Coefficient (A)", min_value=1, max_value=10, value=A)

    run_analysis = st.sidebar.button("🚀 Run Portfolio Optimization")

    return tickers, benchmark_symbol, start_date, end_date, rf_annual, rf, total_capital, A, run_analysis
