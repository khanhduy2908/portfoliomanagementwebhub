import numpy as np
import pandas as pd
import warnings
from numpy.linalg import LinAlgError

def run(adj_returns_combinations, cov_matrix_dict, return_invalid=False):
    valid_combinations = []
    invalid_combinations = []

    for combo in adj_returns_combinations:
        tickers = combo.split('-')

        # --- Lấy kỳ vọng lợi suất ---
        returns_dict = adj_returns_combinations.get(combo, {})
        mu = np.array([returns_dict.get(ticker, np.nan) for ticker in tickers]) / 100  # convert to decimal

        if np.isnan(mu).any() or np.isinf(mu).any():
            invalid_combinations.append((combo, "❌ Kỳ vọng lợi suất chứa NaN hoặc Inf"))
            continue
        if np.all(mu <= 0):
            invalid_combinations.append((combo, "❌ Tất cả kỳ vọng lợi suất ≤ 0"))
            continue

        # --- Lấy ma trận hiệp phương sai ---
        try:
            cov_df = cov_matrix_dict.get(combo)
            if cov_df is None or cov_df.empty:
                raise ValueError("Ma trận hiệp phương sai không tồn tại")
            cov = cov_df.loc[tickers, tickers].values
        except Exception as e:
            invalid_combinations.append((combo, f"❌ Lỗi lấy ma trận hiệp phương sai: {e}"))
            continue

        if np.isnan(cov).any() or np.isinf(cov).any():
            invalid_combinations.append((combo, "❌ Ma trận hiệp phương sai chứa NaN hoặc Inf"))
            continue

        # --- Kiểm tra tính PSD ---
        try:
            eigvals = np.linalg.eigvalsh(cov)
            if np.any(eigvals < -1e-6):
                invalid_combinations.append((combo, "❌ Ma trận hiệp phương sai không PSD"))
                continue
        except LinAlgError as e:
            invalid_combinations.append((combo, f"❌ Lỗi kiểm tra PSD: {e}"))
            continue

        # --- Passed all ---
        valid_combinations.append(combo)

    # --- Thống kê báo cáo ---
    print("\n📊 TỔNG KẾT KIỂM TRA TÍNH KHẢ THI DANH MỤC")
    print("--------------------------------------------------")
    print(f"✅ Số danh mục hợp lệ: {len(valid_combinations)}")
    print(f"❌ Số danh mục không hợp lệ: {len(invalid_combinations)}")

    if invalid_combinations:
        for combo, reason in invalid_combinations:
            warnings.warn(f"[{combo}] {reason}")

    if return_invalid:
        return valid_combinations, invalid_combinations
    return valid_combinations
