# block_a_data.py
import pandas as pd
import numpy as np
import warnings
from vnstock import Vnstock
from itertools import combinations

vnstock_api = Vnstock()

def get_stock_data(symbol, start_date, end_date):
    try:
        df = vnstock_api.stock(symbol=symbol, source="VCI").quote.history(start=start_date, end=end_date)
        if df.empty or 'close' not in df.columns:
            warnings.warn(f"No valid data for {symbol}")
            return pd.DataFrame()
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']].rename(columns=str.capitalize)
        return df
    except Exception as e:
        warnings.warn(f"Failed to load {symbol}: {e}")
        return pd.DataFrame()

def get_first_trading_day(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    return df.groupby(df.index.to_period("M")).apply(lambda x: x.iloc[0]).reset_index(drop=True)

def load_all_monthly_data(tickers, start_date, end_date):
    all_data = []
    for ticker in tickers:
        df = get_stock_data(ticker, start_date, end_date)
        if df.empty:
            continue
        df_month = get_first_trading_day(df)
        df_month['Ticker'] = ticker
        all_data.append(df_month.reset_index(drop=True))
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

def compute_monthly_returns(df):
    df = df.sort_values(['Ticker', 'time'])
    df['Return'] = df.groupby('Ticker')['Close'].pct_change() * 100
    return df.dropna(subset=['Return'])

def run(tickers, benchmark, start_date, end_date):
    data_stocks = load_all_monthly_data(tickers, start_date, end_date)
    data_benchmark = load_all_monthly_data([benchmark], start_date, end_date)

    returns_stocks = compute_monthly_returns(data_stocks)
    returns_benchmark = compute_monthly_returns(data_benchmark)
    returns_benchmark = returns_benchmark[['time', 'Return']].rename(columns={'Return': 'Benchmark_Return'})

    returns_stocks = returns_stocks.merge(returns_benchmark, on='time', how='inner')
    returns_pivot = returns_stocks.pivot(index='time', columns='Ticker', values='Return')
    returns_benchmark.set_index('time', inplace=True)

    combos = list(combinations(tickers, 3))
    labels = ['-'.join(c) for c in combos]

    return data_stocks, data_benchmark, returns_pivot, returns_benchmark, labels
