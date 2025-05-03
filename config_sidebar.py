import streamlit as st
import pandas as pd
import numpy as np
from config import A, rf_annual, total_capital 

# --- SIDEBAR CONFIGURATION ---
def sidebar_config():
    st.sidebar.title("Portfolio Configuration")

    tickers = st.sidebar.text_input("Stock Tickers (comma-separated)", value="VNM,FPT,MWG,VCB,REE")
    tickers = [x.strip().upper() for x in tickers.split(",") if x.strip()]

    benchmark_symbol = st.sidebar.text_input("Benchmark Symbol", value="VNINDEX")

    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))

    rf_annual_user = st.sidebar.number_input("Annual Risk-Free Rate (%)", value=rf_annual * 100.0) / 100
    rf_user = rf_annual_user / 12

    total_capital_user = st.sidebar.number_input("Total Capital (VND)", value=total_capital)

    A_user = st.sidebar.slider("Risk Aversion Coefficient (A)", min_value=1, max_value=10, value=A)

    run_analysis = st.sidebar.button("🚀 Run Portfolio Optimization")

    return tickers, benchmark_symbol, start_date, end_date, rf_annual_user, rf_user, total_capital_user, A_user, run_analysis
