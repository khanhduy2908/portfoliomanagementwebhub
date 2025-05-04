import pandas as pd
import numpy as np
from itertools import combinations
from vnstock import Vnstock

def load_data(tickers, benchmark_symbol, start_date, end_date):
    all_symbols = list(set(tickers + [benchmark_symbol]))
    frames = []
    failed_symbols = []
    tickers_ok = []
    benchmark_ok = False

    client = Vnstock()

    for symbol in all_symbols:
        try:
            df = client.stock_historical_data(symbol=symbol, start=start_date, end=end_date, resolution='1D')
            if df is not None and not df.empty:
                df['Ticker'] = symbol
                frames.append(df)
                if symbol == benchmark_symbol:
                    benchmark_ok = True
                else:
                    tickers_ok.append(symbol)
            else:
                failed_symbols.append(symbol)
        except Exception as e:
            print(f"[ERROR] {symbol}: {e}")
            failed_symbols.append(symbol)

    if not frames:
        raise ValueError("No valid stock data retrieved.")

    df_all = pd.concat(frames, ignore_index=True)
    df_all['time'] = pd.to_datetime(df_all['time'])
    return df_all, tickers_ok, benchmark_ok, failed_symbols

def compute_monthly_return(df):
    if 'Ticker' not in df.columns or 'time' not in df.columns or 'Close' not in df.columns:
        raise KeyError("Missing columns: 'Ticker', 'time' or 'Close'.")

    df = df.sort_values(['Ticker', 'time']).copy()
    df.set_index('time', inplace=True)

    monthly_returns = []
    for ticker in df['Ticker'].unique():
        try:
            close_prices = df[df['Ticker'] == ticker]['Close'].resample('M').last()
            monthly_ret = close_prices.pct_change().dropna()
            monthly_returns.append(pd.DataFrame({ticker: monthly_ret}))
        except Exception as e:
            print(f"Return calc failed for {ticker}: {e}")
            continue

    if not monthly_returns:
        raise ValueError("No valid returns were calculated.")

    return pd.concat(monthly_returns, axis=1)

def compute_benchmark_return(df, benchmark_symbol):
    df_bm = df[df['Ticker'] == benchmark_symbol].copy()
    if df_bm.empty:
        raise ValueError("No benchmark data found.")

    df_bm.set_index('time', inplace=True)
    ret = df_bm['Close'].resample('M').last().pct_change().dropna()
    return ret.to_frame(name='Benchmark_Return')

def run(tickers, benchmark_symbol, start_date, end_date):
    df_all, tickers_ok, benchmark_ok, failed_symbols = load_data(tickers, benchmark_symbol, start_date, end_date)

    if not benchmark_ok:
        raise ValueError("Benchmark symbol failed to load.")

    if not tickers_ok:
        raise ValueError("No valid stock tickers loaded.")

    df_all = df_all[df_all['Ticker'].isin(tickers_ok + [benchmark_symbol])]
    data_stocks = df_all[df_all['Ticker'].isin(tickers_ok)].copy()
    data_benchmark = df_all[df_all['Ticker'] == benchmark_symbol].copy()

    returns_pivot_stocks = compute_monthly_return(data_stocks)
    returns_benchmark = compute_benchmark_return(df_all, benchmark_symbol)

    portfolio_combinations = list(combinations(tickers_ok, 3))

    return data_stocks, data_benchmark, returns_pivot_stocks, returns_benchmark, portfolio_combinations
