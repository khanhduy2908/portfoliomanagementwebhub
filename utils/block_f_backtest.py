# utils/block_f_backtest.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from pytorch_tabnet.tab_model import TabNetRegressor
from collections import defaultdict
import joblib
import os

def run(valid_combinations, features_df, factor_cols, lookback=12, min_samples=100, n_splits=5):
    eval_logs = []
    error_by_stock = defaultdict(list)
    walkforward_results = []

    for combo in valid_combinations:
        subset = combo.split('-')
        df_combo = features_df[features_df['Ticker'].isin(subset)].copy()

        # Tạo dataset dạng chuỗi thời gian
        X_all, y_all, meta = [], [], []
        for ticker in subset:
            df_ticker = df_combo[df_combo['Ticker'] == ticker].sort_values('time')
            for i in range(lookback, len(df_ticker)):
                window = df_ticker[factor_cols].iloc[i - lookback:i].values.flatten()
                target = df_ticker['Return_Close'].iloc[i]
                ts = df_ticker['time'].iloc[i]
                X_all.append(window)
                y_all.append(target)
                meta.append({'time': ts, 'ticker': ticker})

        X_all = np.array(X_all)
        y_all = np.array(y_all)
        meta_df = pd.DataFrame(meta)

        if len(X_all) < min_samples:
            print(f"⚠️ {combo}: Not enough samples ({len(X_all)}). Skipping.")
            continue

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_all)

        split_size = int(len(X_scaled) / (n_splits + 1))
        maes, r2s, accs, dir_accs = [], [], [], []
        preds_all, y_all_vals, tickers_all = [], [], []

        for i in range(n_splits):
            train_end = (i + 1) * split_size
            test_end = train_end + split_size
            if test_end > len(X_scaled): break

            X_train = X_scaled[:train_end]
            y_train = y_all[:train_end].reshape(-1, 1)
            X_test = X_scaled[train_end:test_end]
            y_test = y_all[train_end:test_end].reshape(-1, 1)
            test_meta = meta_df.iloc[train_end:test_end]

            if len(X_test) == 0:
                continue

            model = TabNetRegressor(seed=42)
            model.fit(
                X_train=X_train, y_train=y_train,
                eval_set=[(X_test, y_test)],
                eval_metric=['mae'],
                max_epochs=100, patience=10,
                batch_size=256, virtual_batch_size=128
            )

            preds = model.predict(X_test).squeeze()
            y_true = y_test.squeeze()

            # Metrics
            mae = mean_absolute_error(y_true, preds)
            r2 = r2_score(y_true, preds)
            acc = (np.sign(y_true) == np.sign(preds)).mean()
            dir_acc = ((preds * y_true) > 0).mean()

            maes.append(mae)
            r2s.append(r2)
            accs.append(acc)
            dir_accs.append(dir_acc)

            preds_all.extend(preds)
            y_all_vals.extend(y_true)
            tickers_all.extend(test_meta['ticker'].values)

            # Lưu model từng fold
            os.makedirs("saved_models", exist_ok=True)
            joblib.dump(model, f"saved_models/model_{combo}_fold{i}.pkl")

        if not maes:
            print(f"⚠️ {combo}: No valid folds. Skipping.")
            continue

        walkforward_results.append({
            'Portfolio': combo,
            'MAE': np.mean(maes),
            'R2': np.mean(r2s),
            'Accuracy': np.mean(accs),
            'Directional Accuracy': np.mean(dir_accs)
        })

        error_df = pd.DataFrame({
            'Ticker': tickers_all,
            'True': y_all_vals,
            'Pred': preds_all
        })
        error_df['Error'] = np.abs(error_df['True'] - error_df['Pred'])
        stock_error = error_df.groupby('Ticker')['Error'].mean().sort_values(ascending=False)
        error_by_stock[combo] = stock_error

    if walkforward_results:
        walkforward_df = pd.DataFrame(walkforward_results).sort_values('MAE')
        print("\n📉 Walkforward Evaluation Summary:")
        print(walkforward_df.round(4))

        best_combo = walkforward_df.iloc[0]['Portfolio']
        print(f"\n📋 Stock-level Errors for Best Portfolio ({best_combo}):")
        print(error_by_stock[best_combo].round(4))
    else:
        walkforward_df = pd.DataFrame()
        print("⚠️ No valid portfolios evaluated in Block F.")

    return walkforward_df, error_by_stock
