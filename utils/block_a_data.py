import pandas as pd
import numpy as np
import warnings
from itertools import combinations
from vnstock import Vnstock

def get_first_trading_day(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df.groupby(df.index.to_period('M')).apply(lambda x: x.iloc[0]).reset_index(drop=True)

def get_stock_data(ticker, start, end, source='SSI'):
    try:
        stock = Vnstock().stock(symbol=ticker, source=source)
        df = stock.quote.history(start=start, end=end)
        if df.empty or 'close' not in df.columns:
            return None
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return df
    except Exception as e:
        warnings.warn(f"Error retrieving {ticker}: {e}")
        return None

def load_all_monthly_data(tickers, start, end, source='SSI'):
    stock_data = []
    valid_tickers = []
    for ticker in tickers:
        df = get_stock_data(ticker, start, end, source)
        if df is not None and not df.empty:
            df_monthly = get_first_trading_day(df)
            df_monthly['time'] = df_monthly.index
            df_monthly['Ticker'] = ticker
            stock_data.append(df_monthly.reset_index(drop=True))
            valid_tickers.append(ticker)
    return pd.concat(stock_data, ignore_index=True) if stock_data else pd.DataFrame(), valid_tickers

def compute_monthly_return(df):
    if 'Ticker' not in df.columns or 'Close' not in df.columns or 'time' not in df.columns:
        raise ValueError("Missing required columns in stock data.")
    df = df.sort_values(['Ticker', 'time'])
    df['Return'] = df.groupby('Ticker')['Close'].pct_change() * 100
    return df.dropna(subset=['Return'])

def run(tickers, benchmark_symbol, start_date, end_date):
    # Load data
    df_all, tickers_ok = load_all_monthly_data(tickers, start_date, end_date)
    df_benchmark, benchmark_ok = load_all_monthly_data([benchmark_symbol], start_date, end_date)

    if df_all.empty:
        raise ValueError("No valid stock data retrieved.")
    if df_benchmark.empty:
        raise ValueError("Benchmark symbol has no valid data.")

    # Monthly returns
    returns_stocks = compute_monthly_return(df_all)
    returns_benchmark = compute_monthly_return(df_benchmark)
    returns_benchmark = returns_benchmark[['time', 'Return']].rename(columns={'Return': 'Benchmark_Return'})

    # Merge benchmark
    returns_stocks = returns_stocks.merge(returns_benchmark, on='time', how='inner')

    # Pivot table
    returns_pivot_stocks = returns_stocks.pivot(index='time', columns='Ticker', values='Return')
    returns_benchmark.set_index('time', inplace=True)

    # Portfolio combinations
    portfolio_combinations = list(combinations(tickers_ok, 3))

    return df_all, df_benchmark, returns_pivot_stocks, returns_benchmark, portfolio_combinations
