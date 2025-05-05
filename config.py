# config.py - Centralized Configuration File for Portfolio Optimization System

# --- Investment Parameters ---
tickers = ["VNM", "FPT", "MWG", "VCB", "REE"]
benchmark_symbol = "VNINDEX"
start_date = "2020-01-01"
end_date = None  # None will default to current date in app pipeline
rf_annual = 9  # Annual risk-free rate (%)
rf = rf_annual / 12 / 100  # Monthly risk-free rate
total_capital = 750_000_000  # Total investment capital (VND)
risk_aversion = 5  # Risk aversion coefficient (A)

# --- Portfolio Constraints ---
weight_bounds = (0, 0.4)  # Weight limits per asset
max_assets = 5            # Max number of assets per portfolio

# --- Forecast Model Settings ---
tabnet_params = {
    "n_d": 8,
    "n_a": 8,
    "n_steps": 3,
    "gamma": 1.3,
    "lambda_sparse": 1e-4,
    "optimizer_fn": "adam",
    "scheduler_params": {"mode": "min", "patience": 5, "min_lr": 1e-5},
    "verbose": 0
}
forecast_horizon = 1
feature_cols = ['Volatility', 'Liquidity', 'Momentum', 'Beta']
LOOKBACK_WINDOW = 12 
