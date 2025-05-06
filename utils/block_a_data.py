import pandas as pd
import numpy as np
from vnstock import stock_historical_data

def run(tickers, benchmark_symbol, start_date, end_date):
    # --- Step 1: Download price data ---
    data_all = []
    for ticker in tickers + [benchmark_symbol]:
        try:
            df = stock_historical_data(symbol=ticker, start_date=start_date.strftime('%Y-%m-%d'),
                                       end_date=end_date.strftime('%Y-%m-%d'), resolution='1M', type='stock', source='VCI')
            df['Ticker'] = ticker
            data_all.append(df)
        except Exception as e:
            print(f"[ERROR] Failed to fetch data for {ticker}: {e}")

    if not data_all:
        raise ValueError("❌ No stock data fetched.")

    df_all = pd.concat(data_all, ignore_index=True)

    # --- Step 2: Clean and preprocess ---
    df_all['time'] = pd.to_datetime(df_all['time'])  # ensure datetime format
    df_all = df_all.sort_values(['Ticker', 'time'])

    data_stocks = df_all[df_all['Ticker'].isin(tickers)].copy()
    data_benchmark = df_all[df_all['Ticker'] == benchmark_symbol].copy()

    # --- Step 3: Compute monthly returns ---
    pivot_stocks = data_stocks.pivot(index='time', columns='Ticker', values='Close')
    pivot_benchmark = data_benchmark.pivot(index='time', columns='Ticker', values='Close')

    returns_pivot_stocks = pivot_stocks.pct_change().dropna() * 100
    returns_benchmark = pivot_benchmark.pct_change().dropna() * 100
    returns_benchmark.columns = ['Benchmark_Return']

    # --- Step 4: Ensure DatetimeIndex ---
    if isinstance(returns_pivot_stocks.index, pd.PeriodIndex):
        returns_pivot_stocks.index = returns_pivot_stocks.index.to_timestamp()
    if isinstance(returns_benchmark.index, pd.PeriodIndex):
        returns_benchmark.index = returns_benchmark.index.to_timestamp()

    return data_stocks, data_benchmark, returns_pivot_stocks, returns_benchmark, list(returns_pivot_stocks.columns)
