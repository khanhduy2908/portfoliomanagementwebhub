# utils/block_d_forecast.py

import pandas as pd
import numpy as np
import shap
import optuna
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from pytorch_tabnet.tab_model import TabNetRegressor
from pytorch_tabnet.callbacks import EarlyStopping
import torch
import warnings
import config

warnings.filterwarnings("ignore")

# --- CONFIG ---
lookback = 12
min_samples = 100
feature_cols = ['Return_Close', 'Return_Volume', 'Spread_HL', 'Volatility_Close', 'Ticker_Encoded']

# --- Feature Engineering ---
def engineer_features(df, ticker_id):
    df = df.sort_values('time').copy()
    df['Return_Close'] = df['Close'].pct_change() * 100
    df['Return_Volume'] = df['Volume'].pct_change() * 100
    df['Spread_HL'] = (df['High'] - df['Low']) / df['Low'] * 100
    df['Volatility_Close'] = df['Close'].rolling(window=3).std()
    df['Ticker_Encoded'] = ticker_id
    df.dropna(inplace=True)
    return df

# --- Dataset Construction ---
def construct_dataset(df_combo, subset):
    X, y = [], []
    for ticker in subset:
        df_ticker = df_combo[df_combo['Ticker'] == ticker].sort_values('time')
        for i in range(lookback, len(df_ticker)):
            window = df_ticker[feature_cols].iloc[i-lookback:i].values.flatten()
            target = df_ticker['Return_Close'].iloc[i]
            X.append(window)
            y.append(target)
    return np.array(X), np.array(y)

# --- Train Base + Meta Models ---
def train_stacked_model(X, y, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=False)
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

        model_tab = TabNetRegressor(seed=42)
        model_tab.fit(X_train, y_train.reshape(-1, 1),
                      eval_set=[(X_valid, y_valid.reshape(-1, 1))],
                      eval_metric=['mae'], patience=10, max_epochs=100,
                      batch_size=256, virtual_batch_size=128,
                      callbacks=[EarlyStopping(patience=10)])
        oof_preds_tab.append(model_tab.predict(X_valid).squeeze())

        base_models.append((model_lgb, model_xgb, model_tab))

    # Stack Level-1
    X_meta = np.column_stack((np.concatenate(oof_preds_lgb),
                              np.concatenate(oof_preds_xgb),
                              np.concatenate(oof_preds_tab)))
    y_meta = y[-X_meta.shape[0]:]

    meta_model = RidgeCV(alphas=np.logspace(-3, 3, 7)).fit(X_meta, y_meta)
    return base_models, meta_model

# --- MAIN EXECUTION ---
def run(data_stocks, selected_tickers, selected_combinations):
    adj_returns_combinations = {}
    model_store = {}

    # Step 1: Tạo tập dữ liệu tính năng
    features_all = []
    for i, ticker in enumerate(selected_tickers):
        df = data_stocks[data_stocks['Ticker'] == ticker].copy()
        df_feat = engineer_features(df, ticker_id=i)
        df_feat['Ticker'] = ticker
        features_all.append(df_feat)

    features_df = pd.concat(features_all, ignore_index=True)

    for combo in selected_combinations:
        subset = combo.split('-')
        df_combo = features_df[features_df['Ticker'].isin(subset)].copy()
        X_raw, y = construct_dataset(df_combo, subset)

        if len(X_raw) < min_samples:
            print(f"⚠️ Skipping {combo}: not enough data.")
            continue

        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)

        # Train stacking model
        base_models, meta_model = train_stacked_model(X, y)

        # Final prediction
        pred_lgb = base_models[-1][0].predict(X)
        pred_xgb = base_models[-1][1].predict(X)
        pred_tab = base_models[-1][2].predict(X).squeeze()
        X_stack = np.column_stack([pred_lgb, pred_xgb, pred_tab])
        final_pred = meta_model.predict(X_stack)

        # Trích return dự báo cuối cùng
        adj_return = final_pred[-len(subset):]
        adj_returns_combinations[combo] = dict(zip(subset, adj_return))

        # Save model
        model_store[combo] = {
            'scaler': scaler,
            'base_models': base_models[-1],
            'meta_model': meta_model,
            'features': [f"{col}_t-{t}" for t in reversed(range(lookback)) for col in feature_cols]
        }

        # Optional: SHAP explanation for the last TabNet model
        explainer = shap.Explainer(base_models[-1][2].predict, X)
        shap_values = explainer(X[:50])  # Only a sample
        model_store[combo]['shap_values'] = shap_values

        mae = mean_absolute_error(y[-len(final_pred):], final_pred)
        print(f"✅ {combo} | Final MAE (Stacked): {mae:.4f}")

    return adj_returns_combinations, model_store, features_df
