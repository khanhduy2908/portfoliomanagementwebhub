import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from pytorch_tabnet.tab_model import TabNetRegressor
from collections import defaultdict
import joblib
import os
import hashlib

def run(valid_combinations, features_df, factor_cols, lookback=12, min_samples=100, n_splits=5):
    eval_logs = []
    error_by_stock = defaultdict(list)
    walkforward_results = []

    for combo in valid_combinations:
        subset = combo.split('-')
        df_combo = features_df[features_df['Ticker'].isin(subset)].copy()

        # --- Tạo tập dữ liệu chuỗi thời gian ---
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

        if len(X_all) < min_samples:
            print(f"⚠️ {combo}: Không đủ mẫu ({len(X_all)}). Bỏ qua.")
            continue

        X_all = np.array(X_all)
        y_all = np.array(y_all)
        meta_df = pd.DataFrame(meta)

        # --- Standardize ---
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_all)

        # --- Walkforward Evaluation ---
        split_size = int(len(X_scaled) / (n_splits + 1))
        maes, r2s, accs, dir_accs = [], [], [], []
        preds_all, y_all_vals, tickers_all = [], [], []

        for i in range(n_splits):
            train_end = (i + 1) * split_size
            test_end = train_end + split_size

            if test_end > len(X_scaled):
                break

            X_train = X_scaled[:train_end]
            y_train = y_all[:train_end].reshape(-1, 1)
            X_test = X_scaled[train_end:test_end]
            y_test = y_all[train_end:test_end].reshape(-1, 1)
            test_meta = meta_df.iloc[train_end:test_end]

            if len(X_test) == 0:
                continue

            try:
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

                # --- Save model ---
                os.makedirs("saved_models", exist_ok=True)
                hash_name = hashlib.md5(f"{combo}_fold{i}".encode()).hexdigest()
                joblib.dump(model, f"saved_models/model_{hash_name}.pkl")

            except Exception as e:
                print(f"❌ Fold {i} - Lỗi huấn luyện TabNet cho {combo}: {e}")
                continue

        # --- Kết quả tổng hợp ---
        if not maes:
            print(f"⚠️ {combo}: Không có fold hợp lệ.")
            continue

        walkforward_results.append({
            'Portfolio': combo,
            'MAE': np.mean(maes),
            'R2': np.mean(r2s),
            'Accuracy': np.mean(accs),
            'Directional Accuracy': np.mean(dir_accs)
        })

        # Stock-level error
        error_df = pd.DataFrame({
            'Ticker': tickers_all,
            'True': y_all_vals,
            'Pred': preds_all
        })
        error_df['Error'] = np.abs(error_df['True'] - error_df['Pred'])
        stock_error = error_df.groupby('Ticker')['Error'].mean().sort_values(ascending=False)
        error_by_stock[combo] = stock_error

    # --- In kết quả ---
    walkforward_df = pd.DataFrame(walkforward_results).sort_values('MAE')

    print("\n📉 TỔNG KẾT WALKFORWARD EVALUATION:")
    print(walkforward_df.round(4))

    if not walkforward_df.empty:
        best_combo = walkforward_df.iloc[0]['Portfolio']
        print(f"\n📋 Sai số theo từng cổ phiếu cho danh mục tốt nhất ({best_combo}):")
        print(error_by_stock[best_combo].round(4))

    return walkforward_df, error_by_stock
