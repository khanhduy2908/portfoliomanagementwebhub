import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import optuna
import warnings
import config
from itertools import combinations

# --- A. Compute Factors ---
def compute_factors(data_stocks, returns_benchmark):
    results = []
    for ticker in data_stocks['Ticker'].unique():
        df = data_stocks[data_stocks['Ticker'] == ticker].copy().sort_values('time')
        if df.shape[0] < 6:
            warnings.warn(f"{ticker}: Insufficient data.")
            continue

        df['Return'] = df['Close'].pct_change() * 100
        df['Volatility'] = df['Close'].rolling(window=3).std()
        df['Liquidity'] = df['Volume'].rolling(window=3).mean()
        df['Momentum'] = df['Close'].pct_change(periods=3) * 100
        df.dropna(inplace=True)

        merged = pd.merge(df[['time', 'Return']], returns_benchmark, left_on='time', right_index=True, how='inner')
        if len(merged) < 10:
            warnings.warn(f"{ticker}: Cannot compute Beta – insufficient benchmark data.")
            continue

        model = LinearRegression().fit(merged[['Benchmark_Return']], merged['Return'])
        df = df[df['time'].isin(merged['time'])]
        df['Beta'] = model.coef_[0]
        results.append(df[['time', 'Ticker', 'Return', 'Volatility', 'Liquidity', 'Momentum', 'Beta']])

    if not results:
        raise ValueError("No valid factor data computed.")
    return pd.concat(results, ignore_index=True)

# --- B. Optimize Weights with Optuna ---
def optimize_weights(df):
    def objective(trial):
        weights = np.array([
            trial.suggest_float('w_return', 0, 1),
            trial.suggest_float('w_vol', 0, 1),
            trial.suggest_float('w_liq', 0, 1),
            trial.suggest_float('w_mom', 0, 1),
            trial.suggest_float('w_beta', 0, 1)
        ])
        if weights.sum() == 0:
            return -1e9
        weights /= weights.sum()
        score = (
            weights[0] * df['Return_S'] +
            weights[2] * df['Liquidity_S'] +
            weights[3] * df['Momentum_S'] -
            weights[1] * df['Volatility_S'] -
            weights[4] * df['Beta_S']
        )
        return df.assign(Score=score).nlargest(5, 'Score')['Return'].mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    return study.best_params

# --- C. Apply Weights to Compute Score ---
def apply_weights(df, weights_dict):
    w_arr = np.array(list(weights_dict.values()))
    w_arr /= w_arr.sum()
    df['Score'] = (
        w_arr[0] * df['Return_S'] +
        w_arr[2] * df['Liquidity_S'] +
        w_arr[3] * df['Momentum_S'] -
        w_arr[1] * df['Volatility_S'] -
        w_arr[4] * df['Beta_S']
    )
    df['Rank'] = df['Score'].rank(ascending=False)
    return df

# --- D. Select Stocks based on Strategy ---
def select_stocks(df, strategy):
    features = ['Return_S', 'Volatility_S', 'Liquidity_S', 'Momentum_S', 'Beta_S']
    if df.shape[0] < 3:
        raise ValueError("Not enough stocks for selection.")

    if strategy == 'top5_by_cluster':
        kmeans = KMeans(n_clusters=min(3, df.shape[0]), n_init=10, random_state=42)
        df['Cluster'] = kmeans.fit_predict(df[features])
        selected_df = (
            df.sort_values('Score', ascending=False)
            .groupby('Cluster')
            .head(2)
            .sort_values('Rank')
            .head(5)
            .reset_index(drop=True)
        )
    elif strategy == 'top5_overall':
        selected_df = df.sort_values('Score', ascending=False).head(5).reset_index(drop=True)
    elif strategy == 'strongest_clusters':
        kmeans = KMeans(n_clusters=min(3, df.shape[0]), n_init=10, random_state=42)
        df['Cluster'] = kmeans.fit_predict(df[features])
        cluster_strength = df.groupby('Cluster')['Score'].mean().sort_values(ascending=False)
        strongest = cluster_strength.head(2).index
        selected_df = df[df['Cluster'].isin(strongest)].sort_values('Score', ascending=False).head(5).reset_index(drop=True)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return selected_df

# --- E. Main Run ---
def run(data_stocks, returns_benchmark):
    ranking_df = compute_factors(data_stocks, returns_benchmark)
    latest_month = ranking_df['time'].max()
    latest_data = ranking_df[ranking_df['time'] == latest_month].copy()

    if latest_data.shape[0] < 5:
        raise ValueError("❌ Not enough stocks in latest month for selection.")

    factor_cols = ['Return', 'Volatility', 'Liquidity', 'Momentum', 'Beta']
    scaler = StandardScaler()
    scaled = scaler.fit_transform(latest_data[factor_cols])
    latest_data[[f + '_S' for f in factor_cols]] = scaled

    # Optimize scoring weights
    best_weights = optimize_weights(latest_data)
    config.factor_weights = best_weights

    latest_data = apply_weights(latest_data, best_weights)

    # Select top stocks
    strategy = getattr(config, 'factor_selection_strategy', 'top5_by_cluster')
    selected_df = select_stocks(latest_data.copy(), strategy)

    selected_tickers = selected_df['Ticker'].tolist()
    selected_combinations = list(combinations(selected_tickers, 3))

    return selected_tickers, selected_combinations, latest_data, ranking_df
