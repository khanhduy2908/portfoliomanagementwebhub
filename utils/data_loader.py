# utils/data_loader.py

import pandas as pd
import numpy as np
import warnings
from vnstock import Vnstock
from itertools import combinations

def get_first_trading_day(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df.groupby(df.index.to_period('M')).apply(lambda x: x.iloc[0]).reset_index(drop=True)

def get_stock_data(ticker, start, end):
    try:
        stock = Vnstock().stock(symbol=ticker, source='VCI')
        df = stock.quote.history(start=start, end=end)

        # Kiểm tra dữ liệu
        if df.empty or 'close' not in df.columns:
            warnings.warn(f"{ticker}: Không có dữ liệu hợp lệ.")
            return pd.DataFrame()

        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)

        df = df[['open', 'high', 'low', 'close', 'volume']]
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df['Ticker'] = ticker

        return df

    except Exception as e:
        warnings.warn(f"Lỗi khi tải {ticker}: {e}")
        return pd.DataFrame()

def load_all_monthly_data(tickers, start, end):
    all_data = []
    for ticker in tickers:
        df = get_stock_data(ticker, start, end)
        if not df.empty:
            df_monthly = get_first_trading_day(df)
            df_monthly['Ticker'] = ticker
            df_monthly['time'] = pd.to_datetime(df_monthly.index)
            all_data.append(df_monthly)
    if all_data:
        return pd.concat(all_data).reset_index(drop=True)
    else:
        return pd.DataFrame()

def compute_monthly_return(df):
    if not {'Ticker', 'Close', 'time'}.issubset(df.columns):
        raise ValueError("Thiếu cột bắt buộc: Ticker, Close, time.")
    df = df.sort_values(['Ticker', 'time'])
    df['Return'] = df.groupby('Ticker')['Close'].pct_change() * 100
    return df.dropna(subset=['Return'])

def load_data(tickers, benchmark_symbol, start_date, end_date):
    data_stocks = load_all_monthly_data(tickers, start_date, end_date)
    data_benchmark = load_all_monthly_data([benchmark_symbol], start_date, end_date)

    if data_stocks.empty:
        raise ValueError("❌ Không có dữ liệu cổ phiếu nào được tải thành công.")
    if data_benchmark.empty:
        raise ValueError("❌ Không có dữ liệu benchmark nào được tải thành công.")

    returns_stocks = compute_monthly_return(data_stocks)
    returns_benchmark = compute_monthly_return(data_benchmark)
    returns_benchmark = returns_benchmark[['time', 'Return']].rename(columns={'Return': 'Benchmark_Return'})

    returns_stocks = returns_stocks.merge(returns_benchmark, on='time', how='inner')

    returns_pivot_stocks = returns_stocks.pivot(index='time', columns='Ticker', values='Return')
    returns_benchmark.set_index('time', inplace=True)

    return data_stocks, data_benchmark, returns_pivot_stocks, returns_benchmark
