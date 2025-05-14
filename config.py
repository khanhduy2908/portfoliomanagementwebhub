# === User Inputs from Streamlit Sidebar ===
tickers = []
benchmark_symbol = None

start_date = None
end_date = None

rf_annual = 0.09  # Annual risk-free return (e.g., 9%)
rf = 0.0075       # Monthly risk-free return (e.g., 0.75%)

total_capital = 1_000_000_000
A = 15
risk_score = 25

# === Goal-Based Strategy Matrix ===
time_horizon = "6–10 years"           # e.g. user input
strategy_code = "Strategy 2"          # e.g. Strategy 1–5
alloc_cash = 0.20                     # % of capital to hold in cash
alloc_bond = 0.40                     # % of capital to hold in bonds
alloc_stock = 0.40                    # % of capital to hold in stocks

# === Block B – Factor Selection ===
factor_weights = {}
factor_selection_strategy = "top5_by_cluster"

# === Block C – Covariance Estimation ===
weight_garch = 0.6  # Blend ratio for GARCH in covariance

# === Block D – Return Forecasting ===
lookback = 12
min_samples = 100

# === Block G – HRP + CVaR Optimization ===
alpha_cvar = 0.95
lambda_cvar = 5
beta_l2 = 0.01
cvar_soft_limit = 6.5
n_simulations = 20000

# === Block H – Complete Portfolio Construction ===
y_min = 0.6
y_max = 0.9
min_risk_free_ratio = 0.1  # Constraint: (1 - y) >= min

# Optional: can still be used for additional risk profile logic
max_risk_free_by_profile = {
    'low': 0.85,
    'medium': 0.5,
    'high': 0.2
}

# === Directory paths ===
model_dir = "saved_models"
