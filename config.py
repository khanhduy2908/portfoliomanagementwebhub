# --- Global Config for Portfolio Optimization System ---

# User-selected stock tickers
tickers = []

# Benchmark ticker (e.g., VNINDEX)
benchmark_symbol = None

# Time range
start_date = None
end_date = None

# Risk-free rate
rf_annual = 9.0  # Annual risk-free rate in %
rf = rf_annual / 12 / 100  # Monthly risk-free rate in decimal

# Investment capital (in VND)
total_capital = 750_000_000

# Risk aversion coefficient (A)
A = 15

factor_selection_strategy = "top5_by_cluster"
