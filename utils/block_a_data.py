import pandas as pd
from datetime import datetime
from itertools import combinations
from vnstock import Vnstock


def load_data(tickers, benchmark_symbol, start_date, end_date):
    all_symbols = tickers + [benchmark_symbol]
    data_all = []
    failed_symbols = []

    try:
        stock = Vnstock().stock
    except Exception:
        raise ValueError("Failed to initialize Vnstock().stock")

    for symbol in all_symbols:
        try:
            df = stock.historical_data(symbol, start=start_date, end=end_date, resolution='1D')
            if df is not None and not df.empty:
                df['Ticker'] = symbol
                data_all.append(df)
            else:
                failed_symbols.append(symbol)
        except Exception:
            failed_symbols.append(symbol)

    if not data_all:
        raise ValueError("No valid stock data retrieved.")

    df_all = pd.concat(data_all, ignore_index=True)
    df_all['time'] = pd.to_datetime(df_all['time'])
    return df_all, [s for s in tickers if s not in failed_symbols], benchmark_symbol if benchmark_symbol not in failed_symbols else None, failed_symbols


def compute_monthly_return(df):
    if 'Ticker' not in df.columns or 'time' not in df.columns or 'Close' not in df.columns:
        raise KeyError("Missing required columns")

    df = df.sort_values(['Ticker', 'time']).copy()
    df.set_index('time', inplace=True)
    monthly_returns = []

    for ticker in df['Ticker'].unique():
        try:
            close_prices = df[df['Ticker'] == ticker]['Close'].resample('M').last()
            monthly_ret = close_prices.pct_change().dropna()
            monthly_returns.append(pd.DataFrame({ticker: monthly_ret}))
        except Exception:
            continue

    if not monthly_returns:
        raise ValueError("No monthly returns calculated.")

    return pd.concat(monthly_returns, axis=1)


def compute_benchmark_return(df, benchmark_symbol):
    df_bm = df[df['Ticker'] == benchmark_symbol].copy()
    if df_bm.empty:
        raise ValueError("Benchmark data is missing.")

    df_bm['time'] = pd.to_datetime(df_bm['time'])
    df_bm.set_index('time', inplace=True)
    ret = df_bm['Close'].resample('M').last().pct_change().dropna()
    return ret.to_frame(name='Benchmark_Return')


def run(tickers, benchmark_symbol, start_date, end_date):
    df_all, tickers_ok, benchmark_ok, failed_symbols = load_data(tickers, benchmark_symbol, start_date, end_date)

    if not tickers_ok:
        raise ValueError("No valid stock data retrieved.")

    data_stocks = df_all[df_all['Ticker'].isin(tickers_ok)].copy()
    data_benchmark = df_all[df_all['Ticker'] == benchmark_ok].copy() if benchmark_ok else pd.DataFrame()

    returns_pivot_stocks = compute_monthly_return(data_stocks)
    returns_benchmark = compute_benchmark_return(df_all, benchmark_ok) if benchmark_ok else pd.DataFrame()

    portfolio_combinations = list(combinations(tickers_ok, 3))

    return data_stocks, data_benchmark, returns_pivot_stocks, returns_benchmark, portfolio_combinations
