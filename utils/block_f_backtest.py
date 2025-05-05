# utils/block_f_backtest.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from pytorch_tabnet.tab_model import TabNetRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import joblib
import streamlit as st

# --- CONFIG ---
lookback = 12
n_splits = 5
min_samples = 100

def run(valid_combinations, features_df, factor_cols):
    walkforward_results = []
    error_by_stock = defaultdict(list)

    st.markdown("### 📉 Walkforward Backtest (Model Stability Evaluation)")

    for combo in valid_combinations:
        subset = combo.split('-')
        df_combo = features_df[features_df['Ticker'].isin(subset)].copy()

        # --- Prepare Time Series Dataset ---
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
            st.warning(f"⚠️ `{combo}`: Not enough samples ({len(X_all)}). Skipping.")
            continue

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_all)

        split_size = int(len(X_scaled) / (n_splits + 1))
        maes, r2s, accs, dir_accs = [], [], [], []
        preds_all, y_all_vals, tickers_all = [], [], []

        for i in range(n_splits):
            train_end = (i + 1) * split_size
            test_end = train_end + split_size

            X_train = X_scaled[:train_end]
            y_train = y_all[:train_end].reshape(-1, 1)
            X_test = X_scaled[train_end:test_end]
            y_test = y_all[train_end:test_end].reshape(-1, 1)
            test_meta = meta_df.iloc[train_end:test_end]

            if len(X_test) == 0:
                continue

            model = TabNetRegressor(seed=42)
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                eval_metric=['mae'],
                max_epochs=100,
                patience=10,
                batch_size=256,
                virtual_batch_size=128
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

            joblib.dump(model, f"models/model_{combo}_fold{i}.pkl")

        walkforward_results.append({
            'Portfolio': combo,
            'MAE': np.mean(maes),
            'R2': np.mean(r2s),
            'Accuracy': np.mean(accs),
            'Directional Accuracy': np.mean(dir_accs)
        })

        # Stock-level error tracking
        error_df = pd.DataFrame({
            'Ticker': tickers_all,
            'True': y_all_vals,
            'Pred': preds_all
        })
        error_df['Error'] = np.abs(error_df['True'] - error_df['Pred'])
        stock_error = error_df.groupby('Ticker')['Error'].mean().sort_values(ascending=False)
        error_by_stock[combo] = stock_error

    walkforward_df = pd.DataFrame(walkforward_results).sort_values('MAE')
    st.dataframe(walkforward_df.round(4), use_container_width=True)

    # Visualization: Top metrics
    for metric in ['MAE', 'R2', 'Accuracy']:
        plt.figure(figsize=(10, 5))
        sns.barplot(x='Portfolio', y=metric, data=walkforward_df, palette='crest')
        plt.title(f"{metric} across Portfolios")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()

    return walkforward_df, error_by_stock
