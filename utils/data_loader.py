import pandas as pd
import numpy as np
from itertools import combinations
from vnstock import Vnstock
import warnings

def get_first_trading_day(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df.groupby(df.index.to_period("M")).apply(lambda x: x.iloc[0]).reset_index(drop=True)

def get_stock_data(ticker, start, end):
    try:
        stock = Vnstock().stock(symbol=ticker, source='VCI')
        df = stock.quote.history(start=start, end=end)
        if df.empty or 'close' not in df.columns:
            raise ValueError(f"No valid data for {ticker}")
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return df
    except Exception as e:
        warnings.warn(f"[{ticker}] Error: {e}")
        return pd.DataFrame()

def load_all_monthly_data(tickers, start, end):
    stock_data = []
    error_logs = []
    for ticker in tickers:
        df = get_stock_data(ticker, start, end)
        if df.empty:
            error_logs.append(ticker)
            continue
        df_monthly = get_first_trading_day(df)
        df_monthly['time'] = df_monthly.index
        df_monthly['Ticker'] = ticker
        stock_data.append(df_monthly.reset_index(drop=True))
    return (pd.concat(stock_data, ignore_index=True) if stock_data else pd.DataFrame(), error_logs)

def compute_monthly_return(df):
    if 'Ticker' not in df.columns or 'Close' not in df.columns or 'time' not in df.columns:
        raise ValueError("Missing required columns: Ticker, Close, time")
    df = df.sort_values(['Ticker', 'time'])
    df['Return'] = df.groupby('Ticker')['Close'].pct_change() * 100
    return df.dropna(subset=['Return'])

def load_data(tickers, benchmark_symbol, start_date, end_date):
    # Load stock and benchmark data
    data_stocks, stock_errors = load_all_monthly_data(tickers, start_date, end_date)
    data_benchmark, benchmark_errors = load_all_monthly_data([benchmark_symbol], start_date, end_date)
    error_logs = stock_errors + benchmark_errors

    if data_stocks.empty:
        raise ValueError("❌ Không có dữ liệu cổ phiếu nào được tải thành công.")
    if data_benchmark.empty:
        raise ValueError("❌ Không có dữ liệu benchmark được tải thành công.")

    # Compute returns
    returns_stocks = compute_monthly_return(data_stocks)
    returns_benchmark = compute_monthly_return(data_benchmark)
    returns_benchmark = returns_benchmark[['time', 'Return']].rename(columns={'Return': 'Benchmark_Return'})

    # Merge benchmark return
    returns_stocks = returns_stocks.merge(returns_benchmark, on='time', how='inner')
    returns_pivot_stocks = returns_stocks.pivot(index='time', columns='Ticker', values='Return')
    returns_benchmark.set_index('time', inplace=True)

    return data_stocks, data_benchmark, returns_pivot_stocks, returns_benchmark, error_logs
