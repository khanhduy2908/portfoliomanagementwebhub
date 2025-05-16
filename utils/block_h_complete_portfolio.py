import numpy as np
from scipy.optimize import minimize

def auto_penalty_weight(A, rf, target_alloc):
    """
    Tự động điều chỉnh penalty_weight dựa trên độ bảo thủ của chiến lược.
    """
    stock_weight = target_alloc['stock']
    if stock_weight < 0.25:
        return 1e5  # chiến lược bảo thủ
    elif stock_weight < 0.6:
        return 5e4  # cân bằng
    else:
        return 1e4  # chiến lược tấn công cho phép linh hoạt hơn

def optimize_allocation(best_portfolio, mu, cov, rf, A, total_capital, target_alloc, tolerance=0.03):
    weights_stock_i = np.array([best_portfolio['Weights'][t] for t in best_portfolio['Weights']])
    weights_stock_i /= weights_stock_i.sum()

    def penalty(x, target):
        return np.sum((np.maximum(np.abs(x - target) - tolerance, 0)) ** 2) * penalty_weight

    def utility(x):
        w_cash, w_bond = x
        w_stock = 1 - w_cash - w_bond
        if any(v < 0 or v > 1 for v in [w_cash, w_bond, w_stock]):
            return 1e8
        expected_return = w_stock * np.dot(weights_stock_i, mu) + (w_bond + w_cash) * rf
        volatility = np.sqrt(weights_stock_i.T @ cov @ weights_stock_i) * w_stock
        u = expected_return - 0.5 * A * volatility ** 2
        alloc_vector = np.array([w_cash, w_bond, w_stock])
        target_vector = np.array([target_alloc['cash'], target_alloc['bond'], target_alloc['stock']])
        return -u + penalty(alloc_vector, target_vector)

    x0 = [target_alloc['cash'], target_alloc['bond']]
    bounds = [(0, 1), (0, 1)]
    constraint = {'type': 'eq', 'fun': lambda x: 1 - sum(x) - (1 - sum(x))}

    global penalty_weight
    penalty_weight = auto_penalty_weight(A, rf, target_alloc)

    # Try trust-constr, fallback to SLSQP
    for method in ['trust-constr', 'SLSQP']:
        res = minimize(utility, x0=x0, bounds=bounds, constraints=constraint, method=method)
        if res.success:
            w_cash, w_bond = res.x
            w_stock = 1 - w_cash - w_bond
            return w_cash, w_bond, w_stock

    raise RuntimeError("❌ Allocation optimization failed under both methods")

def get_max_rf_ratio(risk_score, A, alloc_cash, alloc_bond, alloc_stock):
    if 10 <= risk_score <= 17:
        hard_cap = 0.40
    elif 18 <= risk_score <= 27:
        hard_cap = 0.20
    elif 28 <= risk_score <= 40:
        hard_cap = 0.10
    else:
        hard_cap = 0.40

    suggested = 0.02 + (A - 2) * ((0.40 - 0.02) / (25 - 2)) if A <= 25 else 0.40
    max_target_rf = alloc_cash + alloc_bond + 0.4 * alloc_stock
    return min(hard_cap, suggested, max_target_rf)

def run_block_h(
    hrp_result_dict, adj_returns_combinations, cov_matrix_dict,
    rf, A, total_capital, risk_score,
    alloc_cash, alloc_bond, alloc_stock,
    y_min=0.6, y_max=0.9, time_horizon=None,
    tolerance=0.03
):
    if not hrp_result_dict:
        raise ValueError("❌ No valid HRP-CVaR portfolios found.")

    best_key = max(hrp_result_dict, key=lambda k: hrp_result_dict[k]['Sharpe Ratio'])
    best_portfolio = hrp_result_dict[best_key]
    tickers = list(best_portfolio['Weights'].keys())

    mu = np.array([adj_returns_combinations[best_key][t] for t in tickers]) / 100
    cov = cov_matrix_dict[best_key].loc[tickers, tickers].values
    target_alloc = {'cash': alloc_cash, 'bond': alloc_bond, 'stock': alloc_stock}

    w_cash, w_bond, w_stock = optimize_allocation(
        best_portfolio, mu, cov, rf, A, total_capital,
        target_alloc, tolerance
    )

    capital_cash = w_cash * total_capital
    capital_bond = w_bond * total_capital
    capital_stock = w_stock * total_capital
    capital_alloc = {t: w_stock * total_capital * w for t, w in best_portfolio['Weights'].items()}

    weights = np.array([best_portfolio['Weights'][t] for t in tickers])
    weights /= weights.sum()
    mu_p = weights @ mu
    sigma_p = np.sqrt(weights.T @ cov @ weights)
    expected_rc = (capital_stock * mu_p + capital_bond * rf + capital_cash * rf) / total_capital
    sigma_c = (capital_stock * sigma_p) / total_capital
    utility = expected_rc - 0.5 * A * sigma_c ** 2

    max_rf_ratio = get_max_rf_ratio(risk_score, A, alloc_cash, alloc_bond, alloc_stock)
    rf_ratio_now = (capital_cash + capital_bond) / total_capital
    if rf_ratio_now > max_rf_ratio:
        excess = rf_ratio_now - max_rf_ratio
        capital_cash = max(capital_cash - excess * total_capital, 0)
        capital_bond = max(capital_bond - excess * total_capital, 0)
        capital_stock = total_capital - capital_cash - capital_bond
        w_cash, w_bond, w_stock = [capital_cash, capital_bond, capital_stock]
        w_cash /= total_capital
        w_bond /= total_capital
        w_stock /= total_capital

    y_opt = w_stock
    y_capped = min(max(y_min, y_opt), y_max)

    portfolio_info = {
        'portfolio_name': '-'.join(best_key),
        'mu': mu_p,
        'sigma': sigma_p,
        'rf': rf,
        'A': A,
        'expected_rc': expected_rc,
        'sigma_c': sigma_c,
        'utility': utility,
        'capital_cash': capital_cash,
        'capital_bond': capital_bond,
        'capital_stock': capital_stock,
        'actual_cash_ratio': w_cash,
        'actual_bond_ratio': w_bond,
        'actual_stock_ratio': w_stock,
        'target_cash_ratio': alloc_cash,
        'target_bond_ratio': alloc_bond,
        'target_stock_ratio': alloc_stock,
        'capital_rf': capital_cash + capital_bond,
        'capital_risky': capital_stock,
        'y_opt': y_opt,
        'y_capped': y_capped,
        'max_rf_ratio': max_rf_ratio,
        'risk_score': risk_score,
        'time_horizon': time_horizon,
        'tolerance': tolerance
    }

    return (
        best_portfolio, w_stock, capital_alloc, sigma_c, expected_rc,
        weights, tickers, portfolio_info, sigma_p, mu, mu_p, cov,
        w_cash, y_opt, y_capped
    )
