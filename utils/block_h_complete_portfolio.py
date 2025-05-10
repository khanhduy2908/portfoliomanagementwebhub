import numpy as np
from scipy.optimize import minimize_scalar

# === Mapping risk_score & A to realistic risk-free ratio ===
def get_max_rf_ratio(score, A):
    # Hard limit theo risk_score
    if 10 <= score <= 17:
        hard_limit = 0.40
    elif 18 <= score <= 27:
        hard_limit = 0.20
    elif 28 <= score <= 40:
        hard_limit = 0.10
    else:
        raise ValueError("Risk score must be between 10 and 40.")

    # Gợi ý tỷ trọng phi rủi ro theo A (nghịch biến)
    if A >= 25:
        suggested = 0.40
    elif A <= 2:
        suggested = 0.05
    else:
        suggested = 0.05 + (25 - A) * (0.35 / 23)  # từ 0.40 giảm dần về 0.05

    # Kết hợp 2 yếu tố (luôn nằm trong [0, hard_limit])
    return min(hard_limit, suggested)

# === Main Run Function ===
def run(hrp_result_dict, adj_returns_combinations, cov_matrix_dict,
        rf, A, total_capital, risk_score, y_min=0.6, y_max=0.9):

    if not hrp_result_dict:
        raise ValueError("No valid HRP-CVaR results from Block G.")

    # 1. Select optimal risky portfolio
    best_key = max(hrp_result_dict, key=lambda k: hrp_result_dict[k]['Sharpe Ratio'])
    best_portfolio = hrp_result_dict[best_key]
    tickers = list(best_portfolio['Weights'].keys())
    weights = np.array([best_portfolio['Weights'][t] for t in tickers])
    weights /= weights.sum()

    # 2. Calculate mu, sigma_p
    portfolio_name = '-'.join(best_key) if isinstance(best_key, tuple) else str(best_key)
    mu = np.array([adj_returns_combinations[best_key][t] for t in tickers]) / 100
    cov = cov_matrix_dict[best_key].loc[tickers, tickers].values
    sigma_p = np.sqrt(weights.T @ cov @ weights)
    mu_p = weights @ mu

    if mu_p <= 0 or sigma_p <= 0:
        raise ValueError("Selected portfolio has non-positive return or volatility.")

    # 3. Determine max_rf_ratio from both score and A
    max_rf_ratio = get_max_rf_ratio(risk_score, A)
    upper_bound = min(y_max, 1 - max_rf_ratio)
    if upper_bound <= y_min:
        y_min = max(0.01, upper_bound - 0.01)

    # 4. Optimize capital allocation y ∈ [y_min, upper_bound]
    def neg_utility(y):
        expected_rc = y * mu_p + (1 - y) * rf
        sigma_c = y * sigma_p
        return -(expected_rc - 0.5 * A * sigma_c**2)

    result = minimize_scalar(neg_utility, bounds=(y_min, upper_bound), method='bounded')
    y_opt = result.x
    y_capped = np.clip(y_opt, y_min, upper_bound)

    # 5. Capital Allocation
    capital_risky = y_capped * total_capital
    capital_rf = total_capital - capital_risky
    rf_cap_limit = max_rf_ratio * total_capital

    if capital_rf > rf_cap_limit:
        capital_rf = rf_cap_limit
        capital_risky = total_capital - capital_rf
        y_capped = capital_risky / total_capital  # cập nhật lại y

    expected_rc = y_capped * mu_p + (1 - y_capped) * rf
    sigma_c = y_capped * sigma_p
    utility = expected_rc - 0.5 * A * sigma_c**2

    capital_alloc = {t: capital_risky * w for t, w in zip(tickers, weights)}

    portfolio_info = {
        'portfolio_name': portfolio_name,
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
