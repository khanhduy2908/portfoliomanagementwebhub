import os
import pandas as pd
import numpy as np
import warnings
import joblib
import hashlib
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from pytorch_tabnet.tab_model import TabNetRegressor
from pytorch_tabnet.callbacks import EarlyStopping
import torch
import config

warnings.filterwarnings("ignore")

lookback = 12
min_samples = 100
feature_cols = ['Return_Close', 'Return_Volume', 'Spread_HL', 'Volatility_Close', 'Ticker_Encoded']

def engineer_features(df, ticker_id):
    df = df.sort_values('time').copy()
    df['Return_Close'] = df['Close'].pct_change() * 100
    df['Return_Volume'] = df['Volume'].pct_change() * 100
    df['Spread_HL'] = (df['High'] - df['Low']) / df['Low'] * 100
    df['Volatility_Close'] = df['Close'].rolling(window=3).std()
    df['Ticker_Encoded'] = ticker_id
    df.dropna(inplace=True)
    return df

def construct_dataset(df_combo, subset):
    X, y = [], []
    for ticker in subset:
        df_ticker = df_combo[df_combo['Ticker'] == ticker].sort_values('time')
        for i in range(lookback, len(df_ticker)):
            row_window = []
            for lag in reversed(range(lookback)):
                for col in feature_cols:
                    row_window.append(df_ticker[col].iloc[i - lag])
            X.append(row_window)
            y.append(df_ticker['Return_Close'].iloc[i])
    return np.array(X), np.array(y)

def train_stacked_model(X, y, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_preds_lgb, oof_preds_xgb, oof_preds_tab = [], [], []
    base_models = []

    for train_idx, valid_idx in kf.split(X):
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        model_lgb = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05)
        model_lgb.fit(X_train, y_train)
        oof_preds_lgb.append(model_lgb.predict(X_valid))

        model_xgb = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05)
        model_xgb.fit(X_train, y_train)
        oof_preds_xgb.append(model_xgb.predict(X_valid))

        try:
            model_tab = TabNetRegressor(seed=42)
            model_tab.fit(
                X_train, y_train.reshape(-1, 1),
                eval_set=[(X_valid, y_valid.reshape(-1, 1))],
                eval_metric=['mae'],
                patience=10,
                max_epochs=100,
                batch_size=256,
                virtual_batch_size=128,
                callbacks=[EarlyStopping(
                    patience=10,
                    early_stopping_metric='valid_mae',
                    is_maximize=False
                )]
            )
            oof_preds_tab.append(model_tab.predict(X_valid).squeeze())
        except Exception as e:
            model_tab = None
            oof_preds_tab.append(np.zeros_like(y_valid))

        base_models.append((model_lgb, model_xgb, model_tab))

    X_meta = np.column_stack((np.concatenate(oof_preds_lgb),
                              np.concatenate(oof_preds_xgb),
                              np.concatenate(oof_preds_tab)))
    y_meta = y[-X_meta.shape[0]:]
    meta_model = RidgeCV(alphas=np.logspace(-3, 3, 7)).fit(X_meta, y_meta)
    return base_models, meta_model

def run(data_stocks, selected_tickers, selected_combinations):
    adj_returns_combinations = {}
    model_store = {}
    features_all = []
    valid_tickers = []

    for i, ticker in enumerate(selected_tickers):
        df = data_stocks[data_stocks['Ticker'] == ticker].copy()
        df_feat = engineer_features(df, ticker_id=i)

        if df_feat.empty or df_feat.shape[0] < lookback + 5:
            warnings.warn(f"{ticker}: not enough data.")
            continue

        df_feat['Ticker'] = ticker
        features_all.append(df_feat)
        valid_tickers.append(ticker)

    if not features_all:
        raise ValueError("❌ No stock has sufficient data for forecasting.")

    features_df = pd.concat(features_all, ignore_index=True)
    os.makedirs("saved_models", exist_ok=True)
    forecast_valid_combos = []

    for combo in selected_combinations:
        subset = tuple(combo)
        if not all(t in valid_tickers for t in subset):
            continue

        combo_sorted_str = '-'.join(sorted(subset))
        hash_key = hashlib.md5(combo_sorted_str.encode()).hexdigest()
        model_path = f"saved_models/stacked_{hash_key}.pkl"

        if os.path.exists(model_path):
            model_data = joblib.load(model_path)
            adj_returns_combinations[subset] = model_data['adj_return']
            model_store[subset] = model_data['model_store']
            forecast_valid_combos.append(subset)
            continue

        df_combo = features_df[features_df['Ticker'].isin(subset)].copy()
        X_raw, y = construct_dataset(df_combo, subset)

        if len(X_raw) < min_samples:
            warnings.warn(f"⚠️ Skipped {subset}: insufficient samples.")
            continue

        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)
        base_models, meta_model = train_stacked_model(X, y)

        adj_return_dict = {}
        valid_forecast_found = False

        for t in subset:
            df_last = features_df[features_df['Ticker'] == t].sort_values('time').iloc[-lookback:]
            if df_last.shape[0] < lookback:
                continue

            X_last = []
            for lag in reversed(range(lookback)):
                for col in feature_cols:
                    X_last.append(df_last[col].iloc[lag])
            X_last = scaler.transform([X_last])

            pred_lgb = base_models[-1][0].predict(X_last)
            pred_xgb = base_models[-1][1].predict(X_last)
            pred_tab = base_models[-1][2].predict(X_last).squeeze() if base_models[-1][2] else 0
            X_stack = np.column_stack([pred_lgb, pred_xgb, [pred_tab]])
            final = meta_model.predict(X_stack)[0]

            adj_return_dict[t] = final
            valid_forecast_found = True

        if valid_forecast_found:
            model_store_obj = {
                'scaler': scaler,
                'base_models': base_models[-1],
                'meta_model': meta_model,
                'features': [f"{col}_t-{lag}" for lag in reversed(range(lookback)) for col in feature_cols]
            }
            adj_returns_combinations[subset] = adj_return_dict
            model_store[subset] = model_store_obj
            forecast_valid_combos.append(subset)

            joblib.dump({
                'adj_return': adj_return_dict,
                'model_store': model_store_obj
            }, model_path)

    if not adj_returns_combinations:
        raise ValueError("❌ No valid combination has forecast results.")

    config.forecast_valid_combos = forecast_valid_combos
    return adj_returns_combinations, model_store, features_df
