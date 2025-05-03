def load_data(tickers, benchmark_symbol, start_date, end_date):
    import pandas as pd
    from vnstock import Vnstock

    all_symbols = tickers + [benchmark_symbol]
    data_all = []
    for ticker in all_symbols:
        try:
            stock = Vnstock().stock(symbol=ticker, source='VCI')
            df = stock.quote.history(start=start_date, end=end_date)
            if df is not None and not df.empty:
                df['Ticker'] = ticker
                df['time'] = pd.to_datetime(df['time'], errors='coerce')
                df = df.dropna(subset=['time', 'close'])  # Bỏ dòng thiếu dữ liệu
                data_all.append(df)
        except Exception as e:
            print(f"❌ Error fetching {ticker}: {e}")

    if not data_all:
        raise ValueError("❌ Không có dữ liệu cổ phiếu nào được tải thành công.")

    df_all = pd.concat(data_all, ignore_index=True)
    return df_all


def compute_monthly_return(df):
    import pandas as pd

    if 'Ticker' not in df.columns or 'time' not in df.columns or 'close' not in df.columns:
        raise KeyError("❌ Dữ liệu đầu vào thiếu cột cần thiết.")

    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values(['Ticker', 'time'])
    df.set_index('time', inplace=True)

    monthly_returns = []
    for ticker in df['Ticker'].unique():
        df_ticker = df[df['Ticker'] == ticker]['close'].resample('M').last().pct_change().dropna()
        monthly_returns.append(pd.DataFrame({ticker: df_ticker}))

    df_returns = pd.concat(monthly_returns, axis=1)
    return df_returns


def compute_benchmark_return(df, benchmark_symbol):
    import pandas as pd

    df_benchmark = df[df['Ticker'] == benchmark_symbol].copy()
    if df_benchmark.empty:
        raise ValueError("❌ Dữ liệu Benchmark rỗng hoặc không tồn tại.")

    df_benchmark['time'] = pd.to_datetime(df_benchmark['time'])
    df_benchmark.set_index('time', inplace=True)
    df_benchmark = df_benchmark['close'].resample('M').last().pct_change().dropna()
    df_benchmark = df_benchmark.to_frame(name='Benchmark_Return')
    return df_benchmark


def run(tickers, benchmark_symbol, start_date, end_date):
    from itertools import combinations

    df_all = load_data(tickers, benchmark_symbol, start_date, end_date)
    data_stocks = df_all[df_all['Ticker'].isin(tickers)].copy()
    data_benchmark = df_all[df_all['Ticker'] == benchmark_symbol].copy()
    returns_pivot_stocks = compute_monthly_return(data_stocks)
    returns_benchmark = compute_benchmark_return(df_all, benchmark_symbol)
    portfolio_combinations = list(combinations(tickers, 3))
    return data_stocks, data_benchmark, returns_pivot_stocks, returns_benchmark, portfolio_combinations
