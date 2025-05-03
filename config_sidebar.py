import streamlit as st
import pandas as pd
from config import (
    A as A_default,
    rf_annual as rf_annual_default,
    total_capital as capital_default,
    benchmark_symbol as benchmark_default,
    start_date as start_date_default,
    end_date as end_date_default
)

def sidebar_config():
    st.sidebar.header("🔧 Configuration")

    tickers_input = st.sidebar.text_input("Stock Tickers (comma-separated)", value="VNM,FPT,MWG,VCB,REE")
    tickers = [x.strip().upper() for x in tickers_input.split(",") if x.strip()]

    benchmark_symbol = st.sidebar.text_input("Benchmark Symbol", value=benchmark_default)
    
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime(start_date_default))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime(end_date_default))
    
    rf_annual = st.sidebar.number_input("Annual Risk-Free Rate (%)", value=rf_annual_default * 100, step=0.1) / 100
    rf_monthly = rf_annual / 12

    total_capital = st.sidebar.number_input("Total Capital (VND)", value=capital_default, step=50_000_000)
    A = st.sidebar.slider("Risk Aversion Coefficient (A)", min_value=1, max_value=10, value=A_default)

    run_analysis = st.sidebar.button("🚀 Run Portfolio Optimization")

    return {
        "tickers": tickers,
        "benchmark_symbol": benchmark_symbol,
        "start_date": start_date,
        "end_date": end_date,
        "rf_annual": rf_annual,
        "rf": rf_monthly,
        "total_capital": total_capital,
        "A": A,
        "run_analysis": run_analysis
    }