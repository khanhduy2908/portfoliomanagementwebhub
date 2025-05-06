# utils/block_a_data.py

import warnings
import pandas as pd
from vnstock import Vnstock
from itertools import combinations

def run(tickers, benchmark_symbol, start_date, end_date):
    def get_first_trading_day(df):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df.groupby(df.index.to_period('M')).apply(lambda x: x.iloc[0]).reset_index(drop=True)

    def get_stock_data(ticker):
        try:
            stock = Vnstock().stock(symbol=ticker, source='VCI')
            df = stock.quote.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

            if df.empty:
                raise ValueError(f"{ticker}: Empty DataFrame")

            required_cols = {'time', 'close', 'volume'}
            if not required_cols.issubset(df.columns):
                raise ValueError(f"{ticker}: Missing columns {required_cols - set(df.columns)}")

            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            return df
        except Exception as e:
            warnings.warn(f"{ticker}: Data loading failed: {e}")
            return pd.DataFrame()

    def load_all_monthly_data(tickers_list):
        stock_data = []
        for ticker in tickers_list:
            df = get_stock_data(ticker)
            if df.empty or not {'Open', 'High', 'Low', 'Close', 'Volume'}.issubset(df.columns):
                warnings.warn(f"{ticker}: Invalid or incomplete data, skipping.")
                continue
            df_monthly = get_first_trading_day(df)
            if df_monthly.empty:
                warnings.warn(f"{ticker}: No monthly data after filtering.")
                continue
            df_monthly = df_monthly.reset_index(drop=True)
            df_monthly['time'] = pd.to_datetime(df_monthly['time'])
            df_monthly['Ticker'] = ticker
            stock_data.append(df_monthly)
        return pd.concat(stock_data, ignore_index=True) if stock_data else pd.DataFrame()

    data_stocks = load_all_monthly_data(tickers)
    data_benchmark = load_all_monthly_data([benchmark_symbol])

    if data_stocks.empty or data_benchmark.empty:
        raise ValueError("❌ No valid stock or benchmark data is available.")

    def compute_monthly_return(df):
        required = {'Ticker', 'Close', 'time'}
        if not required.issubset(df.columns):
            raise ValueError(f"❌ Missing required columns in input data. Found: {df.columns.tolist()}")
        df = df.sort_values(['Ticker', 'time'])
        df['Return'] = df.groupby('Ticker')['Close'].pct_change() * 100
        return df.dropna(subset=['Return'])

    returns_stocks = compute_monthly_return(data_stocks)
    returns_benchmark = compute_monthly_return(data_benchmark)
    returns_benchmark = returns_benchmark[['time', 'Return']].rename(columns={'Return': 'Benchmark_Return'})

    returns_stocks = returns_stocks.merge(returns_benchmark, on='time', how='inner')
    returns_pivot_stocks = returns_stocks.pivot(index='time', columns='Ticker', values='Return')

    # Ensure datetime index for downstream compatibility
    returns_pivot_stocks.index = pd.to_datetime(returns_pivot_stocks.index)
    returns_benchmark.set_index('time', inplace=True)
    returns_benchmark.index = pd.to_datetime(returns_benchmark.index)

    portfolio_combinations = list(combinations(tickers, 3))
    portfolio_labels = ['-'.join(p) for p in portfolio_combinations]

    return data_stocks, data_benchmark, returns_pivot_stocks, returns_benchmark, portfolio_labels
