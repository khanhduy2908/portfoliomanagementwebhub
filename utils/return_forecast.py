# utils/return_forecast.py

import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import torch

def forecast_returns(combos, tickers, data_stocks):
    adj_returns_combinations = {}
    model_store = {}
    features_all = []

    for ticker in tickers:
        df = data_stocks[data_stocks['Ticker'] == ticker].copy().sort_values('time')
        df['Return_Close'] = df['Close'].pct_change() * 100
        df['Return_Volume'] = df['Volume'].pct_change() * 100
        df['Spread_HL'] = (df['High'] - df['Low']) / df['Low'] * 100
        df['Volatility_Close'] = df['Close'].rolling(window=3).std()
        df['Ticker_Encoded'] = tickers.index(ticker)
        df = df.dropna()
        features_all.append(df)

    features_df = pd.concat(features_all, ignore_index=True)
    feature_cols = ['Return_Close', 'Return_Volume', 'Spread_HL', 'Volatility_Close', 'Ticker_Encoded']
    X = features_df[feature_cols]
    y = features_df['Return_Close'].shift(-1).dropna()
    X = X.loc[y.index]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

    model = TabNetRegressor()
    model.fit(X_train=X_train, y_train=y_train.values.reshape(-1, 1),
              eval_set=[(X_test, y_test.values.reshape(-1, 1))],
              max_epochs=200, patience=10, verbose=0)

    y_pred = model.predict(X_test).flatten()
    mae = mean_absolute_error(y_test, y_pred)

    latest_returns = {}
    for combo in combos:
        tickers_in_combo = combo.split('-')
        latest_returns[combo] = {ticker: float(np.mean(y_pred)) for ticker in tickers_in_combo}

    return latest_returns, model, features_df
