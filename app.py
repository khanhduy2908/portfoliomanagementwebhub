import streamlit as st
import pandas as pd
import sys
import os

# === DYNAMIC PATH FIX FOR IMPORTING LOCAL MODULES ===
APP_DIR = os.path.dirname(__file__)
sys.path.append(APP_DIR)

# === LOCAL IMPORTS ===
from config_sidebar import sidebar_config

# === STREAMLIT PAGE CONFIG ===
st.set_page_config(page_title="📊 Portfolio Optimizer Pro", layout="wide")

# === SIDEBAR CONFIG ===
config = sidebar_config()

# === MAIN HEADER ===
st.title("📈 Institutional-Grade Portfolio Optimization")
st.markdown("A professional-grade investment system built with multi-block architecture, from data to optimization.")

# === CONDITIONAL EXECUTION ===
if config["run_analysis"]:
    st.success("✅ Inputs validated. Running full pipeline...")
    
    # Block A placeholder
    st.subheader("📦 Block A - Load and Prepare Data")
    st.write("Loading stock and benchmark data, computing returns, and generating combinations...")

    # Future: Call block_a_data.run(config) and so on
else:
    st.info("💡 Nhập thông tin bên thanh trái và nhấn **Run Portfolio Optimization** để bắt đầu.")