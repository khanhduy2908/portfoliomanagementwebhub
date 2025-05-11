import numpy as np
from scipy.optimize import minimize_scalar

# --- Tính tỷ lệ risk-free theo khẩu vị rủi ro và mức độ ngại rủi ro A ---
def get_max_rf_ratio(score, A):
    # Hard cap theo risk_score
    if 10 <= score <= 17:
        hard_cap = 0.40
    elif 18 <= score <= 27:
        hard_cap = 0.20
    elif 28 <= score <= 40:
        hard_cap = 0.10
    else:
        raise ValueError("Risk score must be between 10 and 40.")

    # Tính tỷ lệ đề xuất theo A liên tục (mượt)
    if A >= 25:
        suggested = 0.40
    elif A <= 2:
        suggested = 0.02
    else:
        suggested = 0.02 + (A - 2) * ((0.40 - 0.02) / (25 - 2))  # Nội suy tuyến tính

    return min(hard_cap, suggested)

# --- Hàm chính: Tối ưu hóa phân bổ danh mục hoàn chỉnh ---
def run(hrp_result_dict, adj_returns_combinations, cov_matrix_dict,
        rf, A, total_capital, risk_score, y_min=0.6, y_max=0.9):

    if not hrp_result_dict:
        raise ValueError("No valid HRP-CVaR portfolios found.")

    # 1. Chọn danh mục tối ưu theo Sharpe Ratio
    best_key = max(hrp_result_dict, key=lambda k: hrp_result_dict[k]['Sharpe Ratio'])
    best_portfolio = hrp_result_dict[best_key]
    tickers = list(best_portfolio['Weights'].keys())
    weights = np.array([best_portfolio['Weights'][t] for t in tickers])
    weights /= weights.sum()

    # 2. Tính mu và sigma
    mu = np.array([adj_returns_combinations[best_key][t] for t in tickers]) / 100
    cov = cov_matrix_dict[best_key].loc[tickers, tickers].values
    sigma_p = np.sqrt(weights.T @ cov @ weights)
    mu_p = weights @ mu

    if mu_p <= 0 or sigma_p <= 0:
        raise ValueError("Selected portfolio has non-positive return or volatility.")

    # 3. Giới hạn tỷ lệ risk-free
    max_rf_ratio = get_max_rf_ratio(risk_score, A)
    upper_bound = min(y_max, 1 - max_rf_ratio)
    if upper_bound <= y_min:
        y_min = max(0.01, upper_bound - 0.01)

    # 4. Tối ưu hóa Utility
    def neg_utility(y):
        expected_rc = y * mu_p + (1 - y) * rf
        sigma_c = y * sigma_p
        return -(expected_rc - 0.5 * A * sigma_c ** 2)

    result = minimize_scalar(neg_utility, bounds=(y_min, upper_bound), method='bounded')
    y_opt = result.x
    y_capped = np.clip(y_opt, y_min, upper_bound)

    # 5. Phân bổ vốn và kiểm tra ràng buộc risk-free
    capital_risky = y_capped * total_capital
    capital_rf = total_capital - capital_risky
    rf_cap_limit = max_rf_ratio * total_capital

    if capital_rf > rf_cap_limit:
        capital_rf = rf_cap_limit
        capital_risky = total_capital - capital_rf
        y_capped = capital_risky / total_capital

    # 6. Tính thông số danh mục hoàn chỉnh
    expected_rc = y_capped * mu_p + (1 - y_capped) * rf
    sigma_c = y_capped * sigma_p
    utility = expected_rc - 0.5 * A * sigma_c ** 2
    capital_alloc = {t: capital_risky * w for t, w in zip(tickers, weights)}

    # 7. Output
    portfolio_info = {
        'portfolio_name': '-'.join(best_key) if isinstance(best_key, tuple) else str(best_key),
        'mu': mu_p,
        'sigma': sigma_p,
        'rf': rf,
        'y_opt': y_opt,
        'y_capped': y_capped,
        'A': A,
        'expected_rc': expected_rc,
        'sigma_c': sigma_c,
        'utility': utility,
        'capital_risky': capital_risky,
        'capital_rf': capital_rf,
        'risk_score': risk_score,
        'max_rf_ratio': max_rf_ratio
    }

    return (
        best_portfolio, y_capped, capital_alloc,
        sigma_c, expected_rc, weights, tickers,
        portfolio_info, sigma_p, mu, y_opt, mu_p, cov
    )
