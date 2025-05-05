# utils/config.py

# --- Stock selection ---
tickers = []
benchmark_symbol = "VNINDEX"

# --- Time window ---
start_date = None
end_date = None

# --- Risk-free rate ---
rf_annual = 9.0  # % annual
rf = rf_annual / 12 / 100  # monthly rate (as decimal)

# --- Portfolio parameters ---
total_capital = 750_000_000  # in VND
A = 15  # risk aversion coefficient
