import pandas as pd
import numpy as np
from itertools import combinations
import warnings
from datetime import datetime

# --- Import with fallback ---
try:
    from vnstock import stock_historical_data
except ImportError:
    from vnstock import get_price_data as stock_historical_data

def fetch_data(ticker, start_date, end_date):
    try:
        df = stock_historical_data(
            symbol=ticker,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            resolution='1D',
            type='stock',
            source='VCI'
        )
        if df.empty or 'close' not in df.columns:
            raise ValueError(f"No valid data for {ticker}")
        df = df.rename(columns=str.title)
        df['Ticker'] = ticker
        df['time'] = pd.to_datetime(df['Time'])
        return df[['time', 'Ticker', 'Close', 'Volume']]
    except Exception as e:
        warnings.warn(f"❌ Failed to fetch {ticker}: {e}")
        return pd.DataFrame()

def run(tickers, benchmark_symbol, start_date, end_date):
    df_all = []
    for ticker in tickers:
        df = fetch_data(ticker, start_date, end_date)
        if not df.empty:
            df_all.append(df)
    data_stocks = pd.concat(df_all, ignore_index=True) if df_all else pd.DataFrame()

    df_benchmark = fetch_data(benchmark_symbol, start_date, end_date)
    df_benchmark = df_benchmark.rename(columns={'Close': 'Benchmark_Close'})

    # Pivot for return calculation
    pivot_stocks = data_stocks.pivot(index='time', columns='Ticker', values='Close')
    pivot_benchmark = df_benchmark.set_index('time')[['Benchmark_Close']]

    # Calculate monthly returns
    monthly_stocks = pivot_stocks.resample('M').last().pct_change().dropna() * 100
    monthly_benchmark = pivot_benchmark.resample('M').last().pct_change().dropna() * 100
    monthly_benchmark.rename(columns={'Benchmark_Close': 'Benchmark_Return'}, inplace=True)

    # Generate all 3-stock combinations
    portfolio_combinations = list(combinations(tickers, 3))

    return data_stocks, df_benchmark, monthly_stocks, monthly_benchmark, portfolio_combinations
