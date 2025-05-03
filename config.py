# config.py

# === USER-DEFINED PARAMETERS ===
A = 5  # Risk Aversion (Người dùng nhập)
rf_annual = 0.09  # Lãi suất phi rủi ro hàng năm (Người dùng nhập)
total_capital = 750_000_000  # Vốn đầu tư tổng (Người dùng nhập)

# === DERIVED PARAMETERS ===
rf = rf_annual / 12  # Lãi suất phi rủi ro theo tháng

# === PORTFOLIO CONSTRAINTS ===
y_min = 0.6
y_max = 0.9

# === OPTIMIZATION & RISK ===
alpha_cvar = 0.95
lambda_cvar = 10
beta_l2 = 0.05
n_simulations = 30000

# === BENCHMARK CONFIG ===
benchmark_symbol = "VNINDEX"

# === DATE RANGE ===
start_date = "2020-01-01"
