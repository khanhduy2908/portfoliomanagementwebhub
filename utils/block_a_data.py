import pandas as pd
import numpy as np
import warnings
from vnstock import Vnstock
from itertools import combinations

import config  # ✅ import từ thư mục gốc

# --- A.1: Helper - Lấy phiên đầu tháng chuẩn ---
def get_first_trading_day(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df_first = df.groupby(df.index.to_period('M')).apply(lambda x: x.iloc[0])
    df_first.index = df_first.index.get_level_values(1)
    df_first.reset_index(inplace=True)
    df_first.rename(columns={'index': 'time'}, inplace=True)
    return df_first

# --- A.2: Lấy dữ liệu từng cổ phiếu ---
def get_stock_data(ticker, start, end):
    try:
        stock = Vnstock().stock(symbol=ticker, source='VCI')
        df = stock.quote.history(start=start, end=end)
        if df.empty or 'close' not in df.columns:
            warnings.warn(f"{ticker}: không có dữ liệu hợp lệ.")
            return pd.DataFrame()
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return df
    except Exception as e:
        warnings.warn(f"{ticker}: lỗi khi tải dữ liệu - {e}")
        return pd.DataFrame()

# --- A.3: Load toàn bộ dữ liệu theo tháng ---
def load_all_monthly_data(tickers, start, end):
    data_all = []
    for ticker in tickers:
        df = get_stock_data(ticker, start, end)
        if df.empty:
            continue
        df_month = get_first_trading_day(df)
        df_month['Ticker'] = ticker
        data_all.append(df_month)
    if data_all:
        return pd.concat(data_all, ignore_index=True)
    else:
        return pd.DataFrame()

# --- A.4: Tính lợi suất hàng tháng ---
def compute_monthly_return(df):
    required_cols = ['Ticker', 'Close', 'time']
    if not all(col in df.columns for col in required_cols):
        raise ValueError("❌ Thiếu cột cần thiết trong dữ liệu đầu vào.")
    df = df.sort_values(['Ticker', 'time'])
    df['Return'] = df.groupby('Ticker')['Close'].pct_change() * 100
    return df.dropna(subset=['Return'])

# --- A.5: Chạy Block A ---
def run(tickers, benchmark_symbol, start_date, end_date):
    data_stocks = load_all_monthly_data(tickers, start_date, end_date)
    data_benchmark = load_all_monthly_data([benchmark_symbol], start_date, end_date)

    returns_stocks = compute_monthly_return(data_stocks)
    returns_benchmark = compute_monthly_return(data_benchmark)
    returns_benchmark = returns_benchmark[['time', 'Return']].rename(columns={'Return': 'Benchmark_Return'})

    # Merge benchmark để so sánh
    returns_stocks = returns_stocks.merge(returns_benchmark, on='time', how='inner')

    # Pivot bảng lợi suất cổ phiếu
    returns_pivot_stocks = returns_stocks.pivot(index='time', columns='Ticker', values='Return')
    returns_benchmark.set_index('time', inplace=True)

    # Tạo tổ hợp danh mục
    portfolio_combinations = list(combinations(tickers, 3))
    portfolio_labels = ['-'.join(p) for p in portfolio_combinations]

    return data_stocks, data_benchmark, returns_pivot_stocks, returns_benchmark, portfolio_combinations
