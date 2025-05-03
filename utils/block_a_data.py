def load_data(tickers, benchmark_symbol, start_date, end_date):
    import pandas as pd
    from vnstock import stock

    all_symbols = tickers + [benchmark_symbol]
    data_all = []
    for ticker in all_symbols:
        try:
            df = stock.historical_data(ticker, start_date, end_date, resolution='1D')
            if df is not None and not df.empty:
                df['Ticker'] = ticker
                data_all.append(df)
        except Exception as e:
            print(f"❌ Error fetching {ticker}: {e}")

    if not data_all:
        raise ValueError("❌ Không có dữ liệu cổ phiếu nào được tải thành công.")

    df_all = pd.concat(data_all, ignore_index=True)
    if 'time' not in df_all.columns or 'Ticker' not in df_all.columns:
        raise KeyError("❌ Dữ liệu thiếu cột 'Ticker' hoặc 'time'")

    df_all['time'] = pd.to_datetime(df_all['time'])
    return df_all

def compute_monthly_return(df):
    import pandas as pd

    if 'Ticker' not in df.columns or 'time' not in df.columns:
        raise KeyError("❌ Dữ liệu đầu vào thiếu cột 'Ticker' hoặc 'time'")

    df = df.sort_values(['Ticker', 'time'])
    df.set_index('time', inplace=True)

    monthly_returns = []
    for ticker in df['Ticker'].unique():
        df_ticker = df[df['Ticker'] == ticker]['Close'].resample('M').last().pct_change().dropna()
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
    df_benchmark = df_benchmark['Close'].resample('M').last().pct_change().dropna()
    df_benchmark = df_benchmark.to_frame(name='Benchmark_Return')
    return df_benchmark
