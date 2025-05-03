import pandas as pd
from datetime import date

# --- USER INPUT CONFIGURATION (Overwritten by app.py) ---
tickers = ["VNM", "FPT", "MWG", "VCB", "REE"]
benchmark_symbol = "VNINDEX"
start_date = pd.to_datetime("2020-01-01")
end_date = pd.to_datetime("today")

rf_annual = 9.0  # Annual risk-free rate in percent
rf = rf_annual / 12 / 100  # Monthly rate in decimal
total_capital = 750_000_000  # Total capital in VND
A = 5  # Risk aversion coefficient

# --- SYSTEM PARAMETERS ---
min_return_months = 24  # Optional validation parameter
alpha_cvar = 0.95       # CVaR confidence level
cvar_soft_limit = 6.5   # Soft threshold for CVaR constraint
n_simulations = 30000   # Monte Carlo & CVaR simulations (Block G, H, J)

# --- CONSTRAINTS (used in Block G / H) ---
y_min = 0.6             # Min allocation to risky portfolio (CAL line)
y_max = 0.9             # Max allocation to risky portfolio
lambda_cvar = 10        # Penalty coefficient for CVaR
beta_l2 = 0.05          # L2 regularization for robust optimization

# --- MISC ---
solvers = ['SCS', 'ECOS']  # Fallback solvers for CVXPY
lookback = 12              # Feature lookback window
min_samples = 100          # Minimum samples for training
confidence_level = 0.95    # Used in Stress Testing
