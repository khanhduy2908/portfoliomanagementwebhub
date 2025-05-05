import pandas as pd

# --- USER INPUT CONFIGURATION (Overwritten dynamically by app.py) ---
tickers = ["VNM", "FPT", "MWG", "VCB", "REE"]
benchmark_symbol = "VNINDEX"
start_date = pd.to_datetime("2020-01-01")
end_date = pd.to_datetime("today")

rf_annual = 9.0
rf = rf_annual / 12 / 100
total_capital = 750_000_000
A = 15

# --- SYSTEM PARAMETERS ---
min_return_months = 24
lookback = 12
min_samples = 100
confidence_level = 0.95

# --- CVaR & Robust Optimization ---
alpha_cvar = 0.95
lambda_cvar = 10
beta_l2 = 0.05
n_simulations = 30000 

# --- CAL Line Allocation Constraints ---
y_min = 0.6
y_max = 0.9

# --- Solver Settings for cvxpy ---
solvers = ["SCS", "ECOS"]
