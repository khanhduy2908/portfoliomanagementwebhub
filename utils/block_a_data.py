import pandas as pd
import numpy as np
from itertools import combinations
import warnings
import streamlit as st

def load_data(tickers, benchmark_symbol, start_date, end_date):
    from vnstock import stock
    all_symbols = tickers + [benchmark_symbol]
    data_all = []

    st.info(f"🔄 Đang tải dữ liệu cho {len(all_symbols)} mã...")

    for symbol in all_symbols:
        try:
            df = stock.historical_data(symbol, start_date, end_date, resolution='1D')
            if df is not None and not df.empty:
                df['Ticker'] = symbol
                data_all.append(df)
                st.success(f"✅ Đã tải {symbol} ({len(df)} dòng)")
            else:
                st.warning(f"⚠️ Không có dữ liệu cho {symbol}")
        except Exception as e:
            st.error(f"❌ Lỗi khi tải {symbol}: {e}")
            continue

    if not data_all:
        raise ValueError("❌ Không có dữ liệu cổ phiếu nào được tải thành công.")

    df_all = pd.concat(data_all, ignore_index=True)
    if 'time' not in df_all.columns or 'Ticker' not in df_all.columns:
        raise KeyError("❌ Dữ liệu thiếu cột 'time' hoặc 'Ticker'.")

    df_all['time'] = pd.to_datetime(df_all['time'])
    return df_all


def compute_monthly_return(df):
    if 'Ticker' not in df.columns or 'time' not in df.columns or 'Close' not in df.columns:
        raise KeyError("❌ Thiếu cột 'Ticker', 'time' hoặc 'Close' trong dữ liệu.")

    df = df.sort_values(['Ticker', 'time']).copy()
    df.set_index('time', inplace=True)

    monthly_returns = []
    for ticker in df['Ticker'].unique():
        try:
            close_prices = df[df['Ticker'] == ticker]['Close'].resample('M').last()
            monthly_ret = close_prices.pct_change().dropna()
            monthly_returns.append(pd.DataFrame({ticker: monthly_ret}))
        except Exception as e:
            st.warning(f"⚠️ Không tính được return cho {ticker}: {e}")
            continue

    if not monthly_returns:
        raise ValueError("❌ Không có dữ liệu return hợp lệ nào được tính.")

    return pd.concat(monthly_returns, axis=1)


def compute_benchmark_return(df, benchmark_symbol):
    df_bm = df[df['Ticker'] == benchmark_symbol].copy()
    if df_bm.empty:
        raise ValueError("❌ Không có dữ liệu cho benchmark.")

    df_bm['time'] = pd.to_datetime(df_bm['time'])
    df_bm.set_index('time', inplace=True)
    ret = df_bm['Close'].resample('M').last().pct_change().dropna()
    return ret.to_frame(name='Benchmark_Return')


def run(tickers, benchmark_symbol, start_date, end_date):
    st.subheader("📥 Block A – Tải dữ liệu và tính toán return hàng tháng")

    df_all = load_data(tickers, benchmark_symbol, start_date, end_date)
    data_stocks = df_all[df_all['Ticker'].isin(tickers)].copy()
    data_benchmark = df_all[df_all['Ticker'] == benchmark_symbol].copy()

    returns_pivot_stocks = compute_monthly_return(data_stocks)
    returns_benchmark = compute_benchmark_return(df_all, benchmark_symbol)

    portfolio_combinations = list(combinations(tickers, 3))

    st.success("✅ Dữ liệu đã sẵn sàng cho bước tiếp theo.")
    return data_stocks, data_benchmark, returns_pivot_stocks, returns_benchmark, portfolio_combinations
