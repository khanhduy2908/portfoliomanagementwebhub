import numpy as np
from scipy.optimize import minimize_scalar

# --- Tính tỷ lệ tối đa cho risk-free asset dựa theo Risk Score, A và phân bổ mục tiêu ---
def get_max_rf_ratio(score, A, alloc_cash, alloc_bond, alloc_stock):
    if 10 <= score <= 17:
        hard_cap = 0.40
    elif 18 <= score <= 27:
        hard_cap = 0.20
    elif 28 <= score <= 40:
        hard_cap = 0.10
    else:
        raise ValueError("Risk score must be between 10 and 40.")

    if A >= 25:
        suggested = 0.40
    elif A <= 2:
        suggested = 0.02
    else:
        suggested = 0.02 + (A - 2) * ((0.40 - 0.02) / (25 - 2))

    max_target_rf = alloc_cash + alloc_bond + 0.4 * alloc_stock
    return min(hard_cap, suggested, max_target_rf)

# --- Hàm chính: Tối ưu phân bổ danh mục hoàn chỉnh theo y* ---
def run(
    hrp_result_dict, adj_returns_combinations, cov_matrix_dict,
    rf, A, total_capital, risk_score,
    alloc_cash, alloc_bond, alloc_stock,
    y_min=0.6, y_max=0.9):

    if not hrp_result_dict:
        raise ValueError("❌ No valid HRP-CVaR portfolios found.")

    best_key = max(hrp_result_dict, key=lambda k: hrp_result_dict[k]['Sharpe Ratio'])
    best_portfolio = hrp_result_dict[best_key]

    tickers = list(best_portfolio['Weights'].keys())
    weights = np.array([best_portfolio['Weights'][t] for t in tickers])
    weights /= weights.sum()

    mu = np.array([adj_returns_combinations[best_key][t] for t in tickers]) / 100
    cov = cov_matrix_dict[best_key].loc[tickers, tickers].values

    mu_p = weights @ mu
    sigma_p = np.sqrt(weights.T @ cov @ weights)

    if mu_p <= 0 or sigma_p <= 0:
        raise ValueError("❌ Risky portfolio has invalid return or volatility.")

    capital_cash = alloc_cash * total_capital
    capital_bond = alloc_bond * total_capital
    capital_stock = alloc_stock * total_capital

    max_rf_ratio = get_max_rf_ratio(risk_score, A, alloc_cash, alloc_bond, alloc_stock)
    upper_bound = min(y_max, 1 - max_rf_ratio)
    if upper_bound <= y_min:
        y_min = max(0.01, upper_bound - 0.01)

    def neg_utility(y):
        expected_return = y * mu_p + (1 - y) * rf
        volatility = y * sigma_p
        return -(expected_return - 0.5 * A * volatility ** 2)

    result = minimize_scalar(neg_utility, bounds=(y_min, upper_bound), method='bounded')
    y_opt = result.x
    y_capped = np.clip(y_opt, y_min, upper_bound)

    capital_risky = capital_stock * y_capped
    capital_rf_internal = capital_stock * (1 - y_capped)
    capital_rf_total = capital_cash + capital_bond + capital_rf_internal

    rf_cap_limit = max_rf_ratio * total_capital
    if capital_rf_total > rf_cap_limit:
        excess = capital_rf_total - rf_cap_limit
        capital_risky += excess
        capital_rf_total = rf_cap_limit
        y_capped = capital_risky / capital_stock if capital_stock > 0 else 0

    # --- Kiểm tra và điều chỉnh để tránh lệch quá mức so với phân bổ mục tiêu ---
    tolerance = 0.05
    actual_cash_ratio = capital_cash / total_capital
    actual_bond_ratio = capital_bond / total_capital
    actual_stock_ratio = capital_risky / total_capital

    if abs(actual_cash_ratio - alloc_cash) > tolerance or \
       abs(actual_bond_ratio - alloc_bond) > tolerance or \
       abs(actual_stock_ratio - alloc_stock) > tolerance:
        # Tự động hiệu chỉnh lại y nếu lệch quá lớn
        stock_ratio_corrected = alloc_stock
        capital_risky = total_capital * stock_ratio_corrected
        y_capped = capital_risky / capital_stock if capital_stock > 0 else y_capped
        capital_rf_total = total_capital - capital_risky

    expected_rc = (
        capital_stock * (y_capped * mu_p + (1 - y_capped) * rf) +
        capital_bond * rf +
        capital_cash * rf
    ) / total_capital

    sigma_c = (capital_stock * y_capped * sigma_p) / total_capital
    utility = expected_rc - 0.5 * A * sigma_c ** 2

    capital_alloc = {t: capital_risky * w for t, w in zip(tickers, weights)}

    portfolio_info = {
        'portfolio_name': '-'.join(best_key),
        'mu': mu_p,
        'sigma': sigma_p,
        'rf': rf,
        'A': A,
        'risk_score': risk_score,
        'y_opt': y_opt,
        'y_capped': y_capped,
        'expected_rc': expected_rc,
        'sigma_c': sigma_c,
        'utility': utility,
        'capital_risky': capital_risky,
        'capital_rf': capital_rf_total,
        'capital_cash': capital_cash,
        'capital_bond': capital_bond,
        'capital_stock': capital_stock,
        'alloc_cash': alloc_cash,
        'alloc_bond': alloc_bond,
        'alloc_stock': alloc_stock,
        'actual_cash_ratio': actual_cash_ratio,
        'actual_bond_ratio': actual_bond_ratio,
        'actual_stock_ratio': actual_stock_ratio,
        'target_cash_ratio': alloc_cash,
        'target_bond_ratio': alloc_bond,
        'target_stock_ratio': alloc_stock,
        'max_rf_ratio': max_rf_ratio
    }

    return (
        best_portfolio, y_capped, capital_alloc,
        sigma_c, expected_rc, weights, tickers,
        portfolio_info, sigma_p, mu, y_opt, mu_p, cov
    )
