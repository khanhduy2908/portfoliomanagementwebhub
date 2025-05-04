import pandas as pd
import numpy as np
import requests
from datetime import datetime
from itertools import combinations
import streamlit as st


def get_stock_data_ssi(ticker, start_date, end_date):
    try:
        start_ts = int(pd.to_datetime(start_date).timestamp())
        end_ts = int(pd.to_datetime(end_date).timestamp())

        url = f"https://iboard.ssi.com.vn/dchart/api/history?symbol={ticker}&resolution=D&from={start_ts}&to={end_ts}"
        response = requests.get(url)
        data = response.json()

        if 't' not in data or len(data['t']) == 0:
            return pd.DataFrame()

        df = pd.DataFrame({
            'time': pd.to_datetime(data['t'], unit='s'),
            'Open': data['o'],
            'High': data['h'],
            'Low': data['l'],
            'Close': data['c'],
            'Volume': data['v']
        })
        df['Ticker'] = ticker
        return df
    except Exception as e:
        print(f"Error retrieving {ticker}: {e}")
        return pd.DataFrame()


def load_data(tickers, benchmark_symbol, start_date, end_date):
    all_symbols = tickers + [benchmark_symbol]
    all_data = []
    failed_symbols = []

    for symbol in all_symbols:
        df = get_stock_data_ssi(symbol, start_date, end_date)
        if df.empty:
            failed_symbols.append(symbol)
            continue
        all_data.append(df)

    if not all_data:
        raise ValueError("Không có mã cổ phiếu nào có dữ liệu hợp lệ.")

    df_all = pd.concat(all_data, ignore_index=True)
    return df_all, [s for s in tickers if s not in failed_symbols], benchmark_symbol if benchmark_symbol not in failed_symbols else None, failed_symbols


def compute_monthly_return(df):
    if 'Ticker' not in df.columns or 'Close' not in df.columns or 'time' not in df.columns:
        raise ValueError("Thiếu cột cần thiết trong dữ liệu.")

    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values(['Ticker', 'time'])

    monthly_returns = []
    for ticker in df['Ticker'].unique():
        try:
            prices = df[df['Ticker'] == ticker].set_index('time')['Close']
            monthly = prices.resample('M').last().pct_change().dropna()
            monthly_returns.append(pd.DataFrame({ticker: monthly}))
        except Exception as e:
            print(f"Return error {ticker}: {e}")
            continue

    if not monthly_returns:
        raise ValueError("Không có dữ liệu return hợp lệ.")

    return pd.concat(monthly_returns, axis=1)


def compute_benchmark_return(df_all, benchmark_symbol):
    df_bm = df_all[df_all['Ticker'] == benchmark_symbol]
    if df_bm.empty:
        raise ValueError("Không có dữ liệu benchmark.")

    prices = df_bm.set_index('time')['Close']
    monthly = prices.resample('M').last().pct_change().dropna()
    return monthly.to_frame(name="Benchmark_Return")


def run(tickers, benchmark_symbol, start_date, end_date):
    st.subheader("Block A – Data Loading and Return Computation")

    df_all, tickers_ok, benchmark_ok, failed_symbols = load_data(tickers, benchmark_symbol, start_date, end_date)

    if not tickers_ok:
        raise ValueError("Không có mã cổ phiếu nào có dữ liệu hợp lệ.")

    data_stocks = df_all[df_all['Ticker'].isin(tickers_ok)]
    data_benchmark = df_all[df_all['Ticker'] == benchmark_ok] if benchmark_ok else pd.DataFrame()

    returns_pivot_stocks = compute_monthly_return(data_stocks)
    returns_benchmark = compute_benchmark_return(df_all, benchmark_ok) if benchmark_ok else pd.DataFrame()

    portfolio_combinations = list(combinations(tickers_ok, 3))

    st.write("Số mã thành công:", len(tickers_ok))
    if failed_symbols:
        st.warning(f"Các mã lỗi không tải được: {failed_symbols}")

    return data_stocks, data_benchmark, returns_pivot_stocks, returns_benchmark, portfolio_combinations
