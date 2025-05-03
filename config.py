import pandas as pd
from datetime import date

# --- USER INPUT CONFIGURATION (Overwritten by app.py) ---
tickers = ["VNM", "FPT", "MWG", "VCB", "REE"]
benchmark_symbol = "VNINDEX"
start_date = pd.to_datetime("2020-01-01")
end_date = pd.to_datetime("today")

rf_annual = 9.0  # Annual risk-free rate (%)
rf = rf_annual / 12 / 100  # Monthly risk-free rate (decimal)

total_capital = 750_000_000  # Investment capital in VND
A = 5  # Risk aversion coefficient

# --- SYSTEM PARAMETERS ---
min_return_months = 24
alpha_cvar = 0.95
cvar_soft_limit = 6.5
n_simulations = 30000

# --- CAPITAL ALLOCATION CONSTRAINTS ---
y_min = 0.6
y_max = 0.9
lambda_cvar = 10
beta_l2 = 0.05

# --- MACHINE LEARNING / BACKTEST ---
lookback = 12
min_samples = 100

# --- OPTIMIZATION SOLVERS ---
solvers = ['SCS', 'ECOS']

# --- STRESS TEST CONFIG ---
confidence_level = 0.95
