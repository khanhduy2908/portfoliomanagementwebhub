# === User Inputs from Streamlit Sidebar ===
tickers = []
benchmark_symbol = None
start_date = None
end_date = None

rf_annual = 0.09
rf = 0.0075
total_capital = 750_000_000

risk_score = 25           # Risk tolerance score
A = 15                    # Risk aversion coefficient mapped from score
y_min = 0.6               # Minimum leverage, also mapped from score
y_max = 0.9               # Maximum leverage cap

# === Block B – Factor Selection ===
factor_weights = {}
factor_selection_strategy = "top5_by_cluster"

# === Block C – Covariance Estimation ===
weight_garch = 0.6

# === Block D – Return Forecasting ===
lookback = 12
min_samples = 100

# === Block G – HRP + CVaR Optimization ===
alpha_cvar = 0.95
lambda_cvar = 5
beta_l2 = 0.01
cvar_soft_limit = 6.5
n_simulations = 20000

# === Model Directory ===
model_dir = "saved_models"
