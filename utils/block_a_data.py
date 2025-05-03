import pandas as pd
from itertools import combinations
from vnstock import Vnstock

def fetch_data(symbol, start_date, end_date):
    try:
        client = Vnstock()
        df = client.stock_historical_data(symbol=symbol, start=start_date, end=end_date, resolution='1D')
        if df is None or df.empty:
            return None
        df['Ticker'] = symbol
        return df
    except Exception as e:
        print(f"Error retrieving data for {symbol}: {e}")
        return None

def load_data(tickers, benchmark_symbol, start_date, end_date):
    all_symbols = list(set(tickers + [benchmark_symbol]))
    frames = []

    for symbol in all_symbols:
        df = fetch_data(symbol, start_date, end_date)
        if df is not None:
            frames.append(df)
        else:
            print(f"Skipped {symbol} due to error or no data.")

    if not frames:
        raise ValueError("No valid stock data retrieved.")

    df_all = pd.concat(frames, ignore_index=True)
    df_all['time'] = pd.to_datetime(df_all['time'])
    return df_all

def compute_monthly_return(df):
    df = df.copy()
    df = df.sort_values(['Ticker', 'time'])
    df.set_index('time', inplace=True)

    monthly_returns = []
    for ticker in df['Ticker'].unique():
        try:
            series = df[df['Ticker'] == ticker]['Close'].resample('M').last()
            ret = series.pct_change().dropna()
            monthly_returns.append(pd.DataFrame({ticker: ret}))
        except Exception as e:
            print(f"Return calc failed for {ticker}: {e}")
            continue

    if not monthly_returns:
        raise ValueError("No valid returns calculated.")

    return pd.concat(monthly_returns, axis=1)

def compute_benchmark_return(df, benchmark_symbol):
    df_bm = df[df['Ticker'] == benchmark_symbol].copy()
    df_bm.set_index('time', inplace=True)
    bm = df_bm['Close'].resample('M').last().pct_change().dropna()
    return bm.to_frame(name="Benchmark_Return")

def run(tickers, benchmark_symbol, start_date, end_date):
    df_all = load_data(tickers, benchmark_symbol, start_date, end_date)

    data_stocks = df_all[df_all['Ticker'].isin(tickers)].copy()
    data_benchmark = df_all[df_all['Ticker'] == benchmark_symbol].copy()

    returns_pivot_stocks = compute_monthly_return(data_stocks)
    returns_benchmark = compute_benchmark_return(df_all, benchmark_symbol)

    portfolio_combinations = list(combinations(tickers, 3))

    return data_stocks, data_benchmark, returns_pivot_stocks, returns_benchmark, portfolio_combinations
