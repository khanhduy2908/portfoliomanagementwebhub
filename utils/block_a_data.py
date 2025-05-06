import pandas as pd
import numpy as np
import warnings
from vnstock import Vnstock

def fetch_valid_data(symbol, start_date, end_date):
    try:
        df = stock_historical_data(
            symbol=symbol,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            resolution='1M',
            type='stock',
            source='VCI'
        )
        if df is None or df.empty:
            warnings.warn(f"⚠️ {symbol}: No data returned.")
            return None
        df['time'] = pd.to_datetime(df['time'])
        df['Ticker'] = symbol
        return df
    except Exception as e:
        warnings.warn(f"⚠️ {symbol}: Failed to fetch data. Reason: {e}")
        return None

def run(tickers, benchmark_symbol, start_date, end_date):
    stock_data = []
    for ticker in tickers:
        df = fetch_valid_data(ticker, start_date, end_date)
        if df is not None:
            stock_data.append(df)

    if not stock_data:
        raise ValueError("❌ No valid stock data retrieved.")

    data_stocks = pd.concat(stock_data).sort_values(by='time')
    data_stocks['Return'] = data_stocks.groupby('Ticker')['Close'].pct_change() * 100

    # Pivot returns for easier matrix calculation
    returns_pivot_stocks = data_stocks.pivot(index='time', columns='Ticker', values='Return')

    # Benchmark
    df_benchmark = fetch_valid_data(benchmark_symbol, start_date, end_date)
    if df_benchmark is None:
        raise ValueError(f"❌ Failed to retrieve benchmark data: {benchmark_symbol}")
    df_benchmark['Benchmark_Return'] = df_benchmark['Close'].pct_change() * 100
    returns_benchmark = df_benchmark[['time', 'Benchmark_Return']].dropna()
    returns_benchmark.set_index('time', inplace=True)

    return data_stocks, df_benchmark, returns_pivot_stocks, returns_benchmark, list(returns_pivot_stocks.columns)
