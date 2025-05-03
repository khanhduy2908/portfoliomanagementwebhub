import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
from pytorch_tabnet.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import optuna
import warnings


def forecast_returns(selected_combinations, selected_tickers, data_stocks):
    model_store = {}
    adj_returns_combinations = {}
    features_all = []

    for ticker in selected_tickers:
        df = data_stocks[data_stocks['Ticker'] == ticker].copy().sort_values('time')
        df['Return_Close'] = df['Close'].pct_change() * 100
        df['Return_Volume'] = df['Volume'].pct_change() * 100
        df['Spread_HL'] = (df['High'] - df['Low']) / df['Low'] * 100
        df['Volatility_Close'] = df['Close'].rolling(window=3).std()
        df['Ticker_Encoded'] = selected_tickers.index(ticker)
        df = df.dropna()
        features_all.append(df)

    features_df = pd.concat(features_all, ignore_index=True)
    feature_cols = ['Return_Close', 'Return_Volume', 'Spread_HL', 'Volatility_Close', 'Ticker_Encoded']
    target_col = 'Return_Close'

    adj_returns_combinations = {}

    for combo in selected_combinations:
        combo_tickers = combo.split('-')
        combo_df = features_df[features_df['Ticker'].isin(combo_tickers)].copy()

        X = combo_df[feature_cols].values
        y = combo_df[target_col].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = TabNetRegressor()
        model.fit(
            X_train=X_train, y_train=y_train.reshape(-1, 1),
            eval_set=[(X_test, y_test.reshape(-1, 1))],
            max_epochs=200,
            patience=10,
            batch_size=256,
            virtual_batch_size=128,
            drop_last=False,
            callbacks=[EarlyStopping(patience=10)]
        )

        y_pred = model.predict(X_test).reshape(-1)
        mae = mean_absolute_error(y_test, y_pred)

        latest_inputs = features_df[features_df['Ticker'].isin(combo_tickers)].groupby('Ticker').tail(1)
        latest_X = scaler.transform(latest_inputs[feature_cols])
        latest_preds = model.predict(latest_X).reshape(-1)

        adj_returns_combinations[combo] = dict(zip(combo_tickers, latest_preds))
        model_store[combo] = model

    return adj_returns_combinations, model_store, features_df
