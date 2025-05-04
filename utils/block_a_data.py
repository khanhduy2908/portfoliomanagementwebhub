# --- BLOCK A: Data Loading with Multi-Source Fallback ---

import pandas as pd
import numpy as np
import streamlit as st
import warnings
from datetime import datetime
from itertools import combinations

# Optional: try to import vnstock
try:
    from vnstock import Vnstock
    vnstock_available = True
except ImportError:
    vnstock_available = False

# === 1. Fallback Source: CafeF ===
def get_cafef_price(ticker, start_date='2020-01-01'):
    url = f"https://s.cafef.vn/Lich-su-giao-dich-{ticker}.chn"
    try:
        tables = pd.read_html(url)
        df = tables[1].copy()
        df.columns = ['Date', 'Close', 'Change', 'Volume', 'Open', 'High', 'Low']
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        df = df[df['Date'] >= pd.to_datetime(start_date)]
        df.sort_values('Date', inplace=True)
        df['Ticker'] = ticker
        return df
    except Exception as e:
        return pd.DataFrame()

# === 2. Optional Backup CSV (manually uploaded or preloaded) ===
def load_from_backup_csv(ticker):
    path = f"backup_data/{ticker}.csv"
    try:
        df = pd.read_csv(path, parse_dates=['Date'])
        df['Ticker'] = ticker
        return df
    except:
        return pd.DataFrame()

# === 3. Preferred Source: vnstock ===
def get_vnstock_data(ticker, start_date, end_date):
    if not vnstock_available:
        return pd.DataFrame()
    try:
        stock = Vnstock().stock(symbol=ticker, source="SSI")
        df = stock.quote.history(start=start_date, end=end_date)
        df['Date'] = pd.to_datetime(df['time'])
        df = df[['Date', 'close']].rename(columns={'close': 'Close'})
        df['Ticker'] = ticker
        return df
    except:
        return pd.DataFrame()

# === Master Fetch Function ===
def fetch_data(ticker, start_date, end_date):
    df = get_vnstock_data(ticker, start_date, end_date)
    if df.empty:
        df = get_cafef_price(ticker, start_date)
    if df.empty:
        df = load_from_backup_csv(ticker)
    return df

# === Load Data for All Tickers ===
def load_data(tickers, benchmark_symbol, start_date, end_date):
    all_symbols = tickers + [benchmark_symbol]
    all_data = []
    failed = []

    for symbol in all_symbols:
        df = fetch_data(symbol, start_date, end_date)
        if not df.empty:
            all_data.append(df)
        else:
            failed.append(symbol)

    if not all_data:
        raise ValueError("No valid stock data retrieved.")

    df_all = pd.concat(all_data, ignore_index=True)
    return df_all, [s for s in tickers if s not in failed], benchmark_symbol not in failed, failed

# === Compute Monthly Return ===
def compute_monthly_return(df):
    df = df.sort_values(['Ticker', 'Date']).copy()
    df.set_index('Date', inplace=True)
    monthly_returns = []

    for ticker in df['Ticker'].unique():
        prices = df[df['Ticker'] == ticker]['Close'].resample('M').last()
        ret = prices.pct_change().dropna()
        monthly_returns.append(pd.DataFrame({ticker: ret}))

    return pd.concat(monthly_returns, axis=1)

# === Streamlit Block A Runner ===
def run(tickers, benchmark_symbol, start_date, end_date):
    st.subheader("Block A – Data Loading and Return Computation")

    df_all, tickers_ok, benchmark_ok, failed = load_data(tickers, benchmark_symbol, start_date, end_date)
    if len(tickers_ok) == 0:
        raise ValueError("Không có mã cổ phiếu nào có dữ liệu hợp lệ.")

    stock_data = df_all[df_all['Ticker'].isin(tickers_ok)]
    benchmark_data = df_all[df_all['Ticker'] == benchmark_symbol]

    returns_stocks = compute_monthly_return(stock_data)
    returns_benchmark = compute_monthly_return(benchmark_data)
    returns_benchmark = returns_benchmark.rename(columns={returns_benchmark.columns[0]: 'Benchmark_Return'})

    portfolio_combinations = list(combinations(tickers_ok, 3))

    st.write("Sample Stock Returns:")
    st.dataframe(returns_stocks.head())

    return stock_data, benchmark_data, returns_stocks, returns_benchmark, portfolio_combinations
