import pandas as pd
import numpy as np
import warnings
from vnstock import Vnstock
from itertools import combinations
from datetime import datetime, timedelta

# --- Lấy ngày giao dịch đầu tiên mỗi tháng ---
def get_first_trading_day(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df.groupby(df.index.to_period('M')).apply(lambda x: x.iloc[0]).reset_index(drop=True)

# --- Lấy dữ liệu giá cổ phiếu ---
def get_stock_data(ticker, start, end):
    try:
        stock = Vnstock().stock(symbol=ticker, source='TCBS')  # TCBS ổn định hơn VCI
        df = stock.quote.history(start=start, end=end)
        if df.empty or 'close' not in df.columns:
            warnings.warn(f"{ticker} không có dữ liệu hợp lệ.")
            return pd.DataFrame()
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return df
    except Exception as e:
        warnings.warn(f"Lỗi khi tải {ticker}: {e}")
        return pd.DataFrame()

# --- Tải toàn bộ dữ liệu theo tháng ---
def load_all_monthly_data(tickers, start, end):
    stock_data = []
    for ticker in tickers:
        df = get_stock_data(ticker, start, end)
        if df.empty:
            print(f"❌ Skip {ticker} – không có dữ liệu.")
            continue
        df_monthly = get_first_trading_day(df)
        df_monthly['Ticker'] = ticker
        df_monthly['time'] = df_monthly.index
        stock_data.append(df_monthly.reset_index(drop=True))
    if not stock_data:
        return pd.DataFrame()
    return pd.concat(stock_data, ignore_index=True)

# --- Tính return hàng tháng ---
def compute_monthly_return(df):
    if 'Ticker' not in df.columns or 'Close' not in df.columns or 'time' not in df.columns:
        raise ValueError("Thiếu cột cần thiết trong dữ liệu đầu vào.")
    df = df.sort_values(['Ticker', 'time'])
    df['Return'] = df.groupby('Ticker')['Close'].pct_change() * 100
    return df.dropna(subset=['Return'])

# --- Load toàn bộ dữ liệu chuẩn hóa cho app ---
def load_data(ticker_str, benchmark_symbol, start_date, end_date):
    # Chuẩn hóa input
    tickers = [t.strip().upper() for t in ticker_str.split(',') if t.strip()]
    benchmark_symbol = benchmark_symbol.strip().upper()

    # Giảm end_date nếu vượt quá ngày hôm nay
    today = datetime.today()
    if isinstance(end_date, datetime):
        if end_date >= today:
            end_date = today - timedelta(days=2)
    if isinstance(start_date, datetime):
        start_date = start_date.strftime('%Y-%m-%d')
    if isinstance(end_date, datetime):
        end_date = end_date.strftime('%Y-%m-%d')

    # --- Load data ---
    data_stocks = load_all_monthly_data(tickers, start_date, end_date)
    data_benchmark = load_all_monthly_data([benchmark_symbol], start_date, end_date)

    if data_stocks.empty:
        raise ValueError("❌ Không có dữ liệu cổ phiếu nào được tải thành công.")
    if data_benchmark.empty:
        raise ValueError("❌ Không có dữ liệu benchmark được tải thành công.")

    # --- Compute returns ---
    returns_stocks = compute_monthly_return(data_stocks)
    returns_benchmark = compute_monthly_return(data_benchmark)
    returns_benchmark = returns_benchmark[['time', 'Return']].rename(columns={'Return': 'Benchmark_Return'})
    returns_stocks = returns_stocks.merge(returns_benchmark, on='time', how='inner')

    # --- Pivot return theo Ticker ---
    returns_pivot_stocks = returns_stocks.pivot(index='time', columns='Ticker', values='Return')
    returns_benchmark.set_index('time', inplace=True)

    # --- Tổ hợp danh mục ---
    from itertools import combinations
    portfolio_combinations = list(combinations(tickers, 3))
    portfolio_labels = ['-'.join(p) for p in portfolio_combinations]

    return data_stocks, data_benchmark, returns_pivot_stocks, returns_benchmark, portfolio_combinations, portfolio_labels
