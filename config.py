tickers = ["VNM", "FPT", "MWG", "VCB", "REE"]
benchmark_symbol = "VNINDEX"
start_date = "2020-01-01"
end_date = None  # If None, will use current date
rf_annual = 9  # Annual risk-free rate (%)
rf = rf_annual / 12 / 100  # Monthly risk-free rate
total_capital = 750_000_000  # Total investment capital (VND)
risk_aversion = 5  # Coefficient of risk aversion (A)

# --- Portfolio Constraints ---
weight_bounds = (0, 0.4)  # Weight bounds per stock
max_assets = 5

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
forecast_horizon = 1  # Forecast horizon (months)

# --- Evaluation Settings ---
rolling_window = 12  # Rolling window length (months)
LOOKBACK_WINDOW = 12  # Used in Block F (walkforward backtest)
feature_cols = ['Return', 'Volatility', 'Liquidity', 'Momentum', 'Beta']
