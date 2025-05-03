import pandas as pd
import numpy as np
import warnings
from itertools import combinations
from vnstock import Vnstock

# --- Ghi log cảnh báo ---
error_logs = []  # dùng để truyền thông tin lỗi sang app

def get_first_trading_day(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df.groupby(df.index.to_period('M')).apply(lambda x: x.iloc[0]).reset_index(drop=True)

def get_stock_data(ticker, start, end):
    for source in ['VCI', 'TCBS']:
        try:
            stock = Vnstock().stock(symbol=ticker, source=source)
            df = stock.quote.history(start=start, end=end)
            if df.empty or 'close' not in df.columns:
                error_logs.append(f"⚠️ {ticker}: Dữ liệu rỗng từ nguồn {source}")
                continue
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df['Ticker'] = ticker
            return df
        except Exception as e:
            error_logs.append(f"❌ {ticker}: Lỗi khi tải từ {source} - {str(e)}")
            continue
    error_logs.append(f"⛔ {ticker}: Không thể lấy dữ liệu từ cả VCI và TCBS.")
    return pd.DataFrame()

def load_all_monthly_data(tickers, start, end):
    stock_data = []
    for ticker in tickers:
        df = get_stock_data(ticker, start, end)
        if df.empty:
            continue
        df_monthly = get_first_trading_day(df)
        df_monthly['time'] = df_monthly.index
        df_monthly['Ticker'] = ticker
        stock_data.append(df_monthly.reset_index(drop=True))
    return pd.concat(stock_data, ignore_index=True) if stock_data else pd.DataFrame()

def compute_monthly_return(df):
    if 'Ticker' not in df.columns or 'Close' not in df.columns or 'time' not in df.columns:
        raise ValueError("❌ Thiếu cột cần thiết trong dữ liệu đầu vào.")
    df = df.sort_values(['Ticker', 'time'])
    df['Return'] = df.groupby('Ticker')['Close'].pct_change() * 100
    return df.dropna(subset=['Return'])

def load_data(tickers, benchmark_symbol, start_date, end_date):
    global error_logs
    error_logs = []  # reset log mỗi lần load

    data_stocks = load_all_monthly_data(tickers, start_date, end_date)
    data_benchmark = load_all_monthly_data([benchmark_symbol], start_date, end_date)

    if data_stocks.empty:
        raise ValueError("❌ Không có dữ liệu cổ phiếu nào được tải thành công.")
    if data_benchmark.empty:
        raise ValueError("❌ Không có dữ liệu benchmark được tải thành công.")

    returns_stocks = compute_monthly_return(data_stocks)
    returns_benchmark = compute_monthly_return(data_benchmark)
    returns_benchmark = returns_benchmark[['time', 'Return']].rename(columns={'Return': 'Benchmark_Return'})

    returns_stocks = returns_stocks.merge(returns_benchmark, on='time', how='inner')
    returns_pivot_stocks = returns_stocks.pivot(index='time', columns='Ticker', values='Return')
    returns_benchmark.set_index('time', inplace=True)

    portfolio_combinations = list(combinations(tickers, 3))
    portfolio_labels = ['-'.join(p) for p in portfolio_combinations]

    return data_stocks, data_benchmark, returns_pivot_stocks, returns_benchmark, portfolio_combinations, portfolio_labels, error_logs
