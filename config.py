
# config.py - Centralized Configuration File for Portfolio Optimization System

# --- Investment Parameters ---
tickers = ["VNM", "FPT", "MWG", "VCB", "REE"]
benchmark_symbol = "VNINDEX"
start_date = "2020-01-01"
end_date = None  # None sẽ tự động lấy ngày hiện tại trong pipeline
rf_annual = 9  # Lãi suất phi rủi ro hàng năm (%)
rf = rf_annual / 12 / 100  # Lãi suất phi rủi ro hàng tháng (%)
total_capital = 750_000_000  # Tổng vốn đầu tư (VND)
risk_aversion = 5  # Mức độ ngại rủi ro (A)

# --- Portfolio Constraints ---
weight_bounds = (0, 0.4)  # Giới hạn tỷ trọng mỗi cổ phiếu
max_assets = 5            # Số lượng tài sản tối đa trong danh mục

# --- Forecast Model Settings ---
tabnet_params = {
    "n_d": 8,
    "n_a": 8,
    "n_steps": 3,
    "gamma": 1.3,
    "lambda_sparse": 1e-4,
    "optimizer_fn": "adam",
    "scheduler_params": {"mode": "min", "patience": 5, "min_lr": 1e-5},
    "verbose": 0
}
forecast_horizon = 1  # Số tháng dự báo

# --- Evaluation Settings ---
rolling_window = 12  # Số tháng cho phân tích rolling
