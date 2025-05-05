from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import optuna
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
import streamlit as st
import warnings


def run(data_stocks, returns_benchmark):
    def compute_factors(data_stocks, returns_benchmark):
        factor_data = []

        for ticker in data_stocks['Ticker'].unique():
            df = data_stocks[data_stocks['Ticker'] == ticker].copy().sort_values('time')
            if df.shape[0] < 6:
                warnings.warn(f"{ticker}: Not enough data.")
                continue

            df['Return'] = df['Close'].pct_change() * 100
            df['Volatility'] = df['Close'].rolling(window=3).std()
            df['Liquidity'] = df['Volume'].rolling(window=3).mean()
            df['Momentum'] = df['Close'].pct_change(periods=3) * 100
            df.dropna(inplace=True)

            merged = pd.merge(df[['time', 'Return']], returns_benchmark[['Benchmark_Return']],
                              left_on='time', right_index=True, how='inner')

            if len(merged) < 10:
                warnings.warn(f"{ticker}: Not enough data for beta.")
                continue

            X = merged[['Benchmark_Return']]
            y = merged['Return']
            model = LinearRegression().fit(X, y)
            beta = model.coef_[0]
            df = df[df['time'].isin(merged['time'])]
            df['Beta'] = beta
            factor_data.append(df[['time', 'Ticker', 'Return', 'Volatility', 'Liquidity', 'Momentum', 'Beta']])

        return pd.concat(factor_data, ignore_index=True)

    ranking_df = compute_factors(data_stocks, returns_benchmark)
    latest_month = ranking_df['time'].max()
    latest_data = ranking_df[ranking_df['time'] == latest_month].copy()

    scaler = StandardScaler()
    factor_cols = ['Return', 'Volatility', 'Liquidity', 'Momentum', 'Beta']
    scaled_values = scaler.fit_transform(latest_data[factor_cols])
    scaled_df = pd.DataFrame(scaled_values, columns=[f + '_S' for f in factor_cols])
    latest_data = pd.concat([latest_data.reset_index(drop=True), scaled_df], axis=1)

    def objective(trial):
        weights = np.array([
            trial.suggest_float('w_return', 0, 1),
            trial.suggest_float('w_vol', 0, 1),
            trial.suggest_float('w_liq', 0, 1),
            trial.suggest_float('w_mom', 0, 1),
            trial.suggest_float('w_beta', 0, 1)
        ])

        if np.sum(weights) == 0:
            return -1e9

        weights /= np.sum(weights)

        score = (
            weights[0] * latest_data['Return_S'] +
            weights[2] * latest_data['Liquidity_S'] +
            weights[3] * latest_data['Momentum_S'] -
            weights[1] * latest_data['Volatility_S'] -
            weights[4] * latest_data['Beta_S']
        )

        latest_data['Score'] = score
        return latest_data.nlargest(5, 'Score')['Return'].mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    w_opt = study.best_params
    w_arr = np.array(list(w_opt.values()))
    w_arr /= w_arr.sum()

    latest_data['Score'] = (
        w_arr[0] * latest_data['Return_S'] +
        w_arr[2] * latest_data['Liquidity_S'] +
        w_arr[3] * latest_data['Momentum_S'] -
        w_arr[1] * latest_data['Volatility_S'] -
        w_arr[4] * latest_data['Beta_S']
    )
    latest_data['Rank'] = latest_data['Score'].rank(ascending=False)

    features_for_cluster = [f + '_S' for f in factor_cols]
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    latest_data['Cluster'] = kmeans.fit_predict(latest_data[features_for_cluster])

    selected_df = (
        latest_data.sort_values('Score', ascending=False)
        .groupby('Cluster')
        .head(2)
        .sort_values('Rank')
        .head(5)
        .reset_index(drop=True)
    )

    selected_tickers = selected_df['Ticker'].tolist()
    selected_combinations = ['-'.join(c) for c in combinations(selected_tickers, 3)]

    st.subheader("Top 5 Stocks by Factor Ranking")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=selected_df, x='Ticker', y='Score', palette='Blues_d', edgecolor='black', ax=ax)
    ax.set_title("Top 5 Stocks by Composite Score", fontsize=14)
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Composite Score")
    st.pyplot(fig)

    return selected_tickers, selected_combinations, latest_data
