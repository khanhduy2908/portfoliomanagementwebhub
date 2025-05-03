import pandas as pd
import streamlit as st
from vnstock import stock
from itertools import combinations


def load_data(tickers, benchmark_symbol, start_date, end_date):
    all_symbols = tickers + [benchmark_symbol]
    data_all = []

    st.subheader("Block A – Data Loading and Monthly Return Computation")
    st.info(f"Loading data for {len(all_symbols)} tickers...")

    for symbol in all_symbols:
        try:
            df = stock.historical_data(symbol, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), resolution='1D')
            if df is not None and not df.empty:
                df['Ticker'] = symbol
                data_all.append(df)
                st.success(f"{symbol}: {len(df)} rows retrieved.")
            else:
                st.warning(f"No data for {symbol}")
        except Exception as e:
            st.error(f"Error retrieving {symbol}: {e}")
            continue

    if not data_all:
        raise ValueError("No stock data was successfully retrieved.")

    df_all = pd.concat(data_all, ignore_index=True)
    df_all['time'] = pd.to_datetime(df_all['time'])
    return df_all


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
            st.warning(f"Return calc failed for {ticker}: {e}")
            continue

    if not monthly_returns:
        raise ValueError("No valid returns were calculated.")

    return pd.concat(monthly_returns, axis=1)


def compute_benchmark_return(df, benchmark_symbol):
    df_bm = df[df['Ticker'] == benchmark_symbol].copy()
    if df_bm.empty:
        raise ValueError("No benchmark data found.")

    df_bm['time'] = pd.to_datetime(df_bm['time'])
    df_bm.set_index('time', inplace=True)
    ret = df_bm['Close'].resample('M').last().pct_change().dropna()
    return ret.to_frame(name='Benchmark_Return')


def run(tickers, benchmark_symbol, start_date, end_date):
    df_all = load_data(tickers, benchmark_symbol, start_date, end_date)

    data_stocks = df_all[df_all['Ticker'].isin(tickers)].copy()
    data_benchmark = df_all[df_all['Ticker'] == benchmark_symbol].copy()

    returns_pivot_stocks = compute_monthly_return(data_stocks)
    returns_benchmark = compute_benchmark_return(df_all, benchmark_symbol)

    portfolio_combinations = list(combinations(tickers, 3))

    st.success("Block A completed: Data loaded & returns calculated.")
    return data_stocks, data_benchmark, returns_pivot_stocks, returns_benchmark, portfolio_combinations
