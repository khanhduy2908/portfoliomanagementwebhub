import pandas as pd
import numpy as np
import streamlit as st
from vnstock import Vnstock
from itertools import combinations

def load_data(tickers, benchmark_symbol, start_date, end_date):
    stock = Vnstock().stock
    all_symbols = tickers + [benchmark_symbol]
    data_all = []

    st.info(f"Loading data for {len(all_symbols)} tickers...")

    for symbol in all_symbols:
        try:
            df = stock.historical_data(symbol, start=start_date, end=end_date, resolution='1D')
            if df is not None and not df.empty:
                df['Ticker'] = symbol
                data_all.append(df)
                st.write(f"✔ {symbol}: {len(df)} rows")
            else:
                st.warning(f"⚠ No data for {symbol}")
        except Exception as e:
            st.error(f"Error retrieving {symbol}: {e}")

    if not data_all:
        raise ValueError("No stock data was successfully retrieved.")

    df_all = pd.concat(data_all, ignore_index=True)
    df_all['time'] = pd.to_datetime(df_all['time'])
    return df_all


def compute_monthly_return(df):
    df = df.sort_values(['Ticker', 'time']).copy()
    df.set_index('time', inplace=True)

    monthly_returns = {}
    for ticker in df['Ticker'].unique():
        try:
            close_prices = df[df['Ticker'] == ticker]['Close'].resample('M').last()
            monthly_ret = close_prices.pct_change().dropna()
            monthly_returns[ticker] = monthly_ret
        except Exception as e:
            st.warning(f"Return calculation failed for {ticker}: {e}")

    if not monthly_returns:
        raise ValueError("No valid returns calculated.")

    return pd.DataFrame(monthly_returns)


def compute_benchmark_return(df, benchmark_symbol):
    df_bm = df[df['Ticker'] == benchmark_symbol].copy()
    if df_bm.empty:
        raise ValueError("No benchmark data found.")

    df_bm['time'] = pd.to_datetime(df_bm['time'])
    df_bm.set_index('time', inplace=True)
    returns = df_bm['Close'].resample('M').last().pct_change().dropna()
    return returns.to_frame(name='Benchmark_Return')


def run(tickers, benchmark_symbol, start_date, end_date):
    st.subheader("Block A – Load Data & Compute Monthly Returns")

    df_all = load_data(tickers, benchmark_symbol, start_date, end_date)
    data_stocks = df_all[df_all['Ticker'].isin(tickers)].copy()
    data_benchmark = df_all[df_all['Ticker'] == benchmark_symbol].copy()

    returns_pivot_stocks = compute_monthly_return(data_stocks)
    returns_benchmark = compute_benchmark_return(df_all, benchmark_symbol)

    portfolio_combinations = list(combinations(tickers, 3))

    st.success("Block A completed: Data and returns ready.")
    return data_stocks, data_benchmark, returns_pivot_stocks, returns_benchmark, portfolio_combinations
