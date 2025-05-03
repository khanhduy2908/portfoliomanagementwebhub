import pandas as pd
import numpy as np
import torch
import warnings
from pytorch_tabnet.tab_model import TabNetRegressor
from pytorch_tabnet.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import optuna
from collections import defaultdict

# === BLOCK B.1: Feature Engineering ===
def prepare_features(data_stocks, tickers):
    features_all = []
    for ticker in tickers:
        df = data_stocks[data_stocks['Ticker'] == ticker].copy()
        df = df.sort_values('time')
        df['Return_Close'] = df['Close'].pct_change() * 100
        df['Return_Volume'] = df['Volume'].pct_change() * 100
        df['Spread_HL'] = (df['High'] - df['Low']) / df['Low'] * 100
        df['Volatility_Close'] = df['Close'].rolling(window=3).std()
        df['Ticker_Encoded'] = tickers.index(ticker)
        df['Target'] = df['Close'].shift(-1).pct_change(periods=1) * 100
        df = df.dropna()
        features_all.append(df)
    features_df = pd.concat(features_all, ignore_index=True)
    return features_df

# === BLOCK B.2: Forecasting Returns with TabNet ===
def train_tabnet(X_train, y_train, X_valid, y_valid, params=None):
    if params is None:
        params = {
            'n_d': 8, 'n_a': 8, 'n_steps': 3,
            'gamma': 1.3, 'lambda_sparse': 1e-4,
            'optimizer_fn': torch.optim.Adam,
            'optimizer_params': dict(lr=2e-2),
            'mask_type': 'entmax'
        }
    model = TabNetRegressor(**params)

    early_stop_cb = EarlyStopping(
        early_stopping_metric="valid_loss",
        is_maximize=False,
        patience=10
    )

    model.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_valid, y_valid)],
        eval_name=["valid"],
        eval_metric=["rmse"],
        max_epochs=200,
        patience=20,
        batch_size=128,
        virtual_batch_size=64,
        callbacks=[early_stop_cb]
    )

    return model

def forecast_returns(selected_combinations, selected_tickers, data_stocks):
    adj_returns_combinations = {}
    model_store = {}
    features_df = prepare_features(data_stocks, selected_tickers)

    for combo in selected_combinations:
        tickers = combo if isinstance(combo, tuple) else tuple(combo.split('-'))
        combo_df = features_df[features_df['Ticker'].isin(tickers)].copy()
        X = combo_df[['Return_Close', 'Return_Volume', 'Spread_HL', 'Volatility_Close', 'Ticker_Encoded']]
        y = combo_df['Target']

        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)

        model = train_tabnet(X_train_scaled, y_train.values, X_valid_scaled, y_valid.values)
        model_store["-".join(tickers)] = model

        pred_returns = {}
        for t in tickers:
            X_pred = features_df[features_df['Ticker'] == t].copy()
            X_pred = X_pred[['Return_Close', 'Return_Volume', 'Spread_HL', 'Volatility_Close', 'Ticker_Encoded']]
            X_pred_scaled = scaler.transform(X_pred)
            y_pred = model.predict(X_pred_scaled)
            pred_returns[t] = np.mean(y_pred)

        adj_returns_combinations["-".join(tickers)] = pred_returns

    return adj_returns_combinations, model_store, features_df

# === BLOCK B.3 (Optional): Walkforward Backtesting Placeholder ===
def walkforward_evaluation(valid_combinations, features_df):
    walkforward_df = pd.DataFrame()  # Future enhancement placeholder
    best_combo = list(valid_combinations)[0]
    best_weights = [1/3] * 3
    error_by_stock = {}
    return walkforward_df, best_combo, best_weights, error_by_stock
