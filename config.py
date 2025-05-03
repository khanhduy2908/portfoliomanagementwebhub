# === USER-DEFINED DEFAULTS (dùng trong sidebar, có thể override) ===
import pandas as pd

DEFAULT_TICKERS = ["VNM", "FPT", "MWG", "VCB", "REE"]
DEFAULT_BENCHMARK = "VNINDEX"
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = pd.Timestamp.today().strftime('%Y-%m-%d')

DEFAULT_RF_ANNUAL = 9.0  # %
DEFAULT_TOTAL_CAPITAL = 750_000_000
DEFAULT_RISK_AVERSION = 5

# === DERIVED PARAMETERS ===
rf = DEFAULT_RF_ANNUAL / 100 / 12  # Monthly risk-free rate

# === PORTFOLIO CONSTRAINTS ===
y_min = 0.6
y_max = 0.9

# === OPTIMIZATION & RISK ===
alpha_cvar = 0.95
lambda_cvar = 10
beta_l2 = 0.05
n_simulations = 30000
