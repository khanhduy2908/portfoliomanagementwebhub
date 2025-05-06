# config.py

# === User Inputs from Streamlit Sidebar ===
tickers = []                # List of stock tickers selected by user
benchmark_symbol = None     # Benchmark index symbol

start_date = None           # Analysis start date (datetime)
end_date = None             # Analysis end date (datetime)

rf_annual = 0.09            # Annual risk-free rate (default 9%)
rf = 0.0075                 # Monthly risk-free rate

total_capital = 750_000_000  # Capital in VND
A = 15                      # Risk aversion coefficient

# === Block B – Factor Selection ===
factor_weights = {}         # Optimal weights for ranking factors

# === Block C – Covariance Estimation ===
weight_garch = 0.6          # GARCH vs Ledoit-Wolf blend ratio (0.0 – 1.0)

# === Block D – Return Forecasting ===
lookback = 12               # Number of periods (months) for feature window
min_samples = 100           # Minimum number of samples per portfolio for training

# === Block G – HRP + CVaR Optimization ===
alpha_cvar = 0.95           # Confidence level for CVaR
lambda_cvar = 5             # Penalization weight for CVaR
beta_l2 = 0.01              # L2 regularization term
cvar_soft_limit = 6.5       # Soft cap for CVaR (%)
n_simulations = 20000       # Monte Carlo scenarios

# === Block H – Complete Portfolio Construction ===
y_min = 0.6                 # Minimum leverage cap
y_max = 0.9                 # Maximum leverage cap

# === Directories for Model Persistence ===
model_dir = "saved_models" # Directory where ML models are saved

factor_selection_strategy = "top5_by_cluster"
# config.py

def map_risk_score_to_A(score: int) -> int:
    """
    Map user risk score (10–40) to a risk aversion coefficient A (integer).
    Higher score → lower A → higher risk tolerance.
    """
    if score <= 17:
        return 30  # Very risk averse
    elif 18 <= score <= 27:
        return 20  # Moderate risk aversion
    elif 28 <= score <= 40:
        return 10  # Low risk aversion
    else:
        raise ValueError("Risk score must be between 10 and 40")

def get_risk_profile_description(score: int) -> str:
    """
    Provide a short description of the user's risk tolerance based on score.
    """
    if score <= 17:
        return "Low risk tolerance – Prioritize capital preservation (bonds, deposits)."
    elif 18 <= score <= 27:
        return "Moderate risk tolerance – Balanced portfolio of stocks and bonds."
    elif 28 <= score <= 40:
        return "High risk tolerance – Focus on growth stocks and high-return assets."
    else:
        return "Invalid score"
