import numpy as np
from scipy.optimize import minimize

def optimize_allocation(
    best_portfolio, mu, cov, rf, A, total_capital,
    target_alloc, margin=0.03, epsilon=1e-6
):
    """
    Tối ưu tỷ trọng phân bổ (cash, bond, stock) với ràng buộc target ± margin,
    xử lý ma trận covariance gần suy biến, tránh lỗi singular matrix.
    """

    weights_stock_i = np.array([best_portfolio['Weights'][t] for t in best_portfolio['Weights']])
    weights_stock_i /= weights_stock_i.sum()

    # Regularize covariance matrix nếu gần suy biến
    cond_number = np.linalg.cond(cov)
    if cond_number > 1e10:
        cov = cov + np.eye(cov.shape[0]) * epsilon

    def utility(x):
        w_cash, w_bond = x
        w_stock = 1 - w_cash - w_bond

        # Ràng buộc trong hàm mục tiêu dưới dạng penalty:
        penalty = 0
        if w_stock < 0 or w_cash < 0 or w_bond < 0 or w_cash > 1 or w_bond > 1:
            penalty += 1e6

        # Penalty nếu lệch khỏi margin
        penalty += 1e5 * max(0, abs(w_cash - target_alloc['cash']) - margin) ** 2
        penalty += 1e5 * max(0, abs(w_bond - target_alloc['bond']) - margin) ** 2
        penalty += 1e5 * max(0, abs(w_stock - target_alloc['stock']) - margin) ** 2

        expected_return = w_stock * np.dot(weights_stock_i, mu) + (w_bond + w_cash) * rf
        volatility = np.sqrt(weights_stock_i.T @ cov @ weights_stock_i) * w_stock
        u = expected_return - 0.5 * A * volatility ** 2
        return -u + penalty

    # Ràng buộc bất đẳng thức: tổng cash + bond ≤ 1 (stock = 1 - cash - bond ≥ 0)
    constraints = ({
        'type': 'ineq',
        'fun': lambda x: 1 - (x[0] + x[1])
    })

    # Bounds cho cash và bond theo margin, đảm bảo nằm trong [0,1]
    bounds = [
        (max(0, target_alloc['cash'] - margin), min(1, target_alloc['cash'] + margin)),
        (max(0, target_alloc['bond'] - margin), min(1, target_alloc['bond'] + margin))
    ]

    # Khởi tạo gần target
    initial_guess = [target_alloc['cash'], target_alloc['bond']]

    result = minimize(utility, x0=initial_guess, bounds=bounds, constraints=constraints, method='SLSQP')

    if not result.success:
        # fallback: trả về tỷ lệ target không thay đổi nếu tối ưu không thành công
        w_cash_opt = target_alloc['cash']
        w_bond_opt = target_alloc['bond']
        w_stock_opt = 1 - w_cash_opt - w_bond_opt
    else:
        w_cash_opt, w_bond_opt = result.x
        w_stock_opt = 1 - w_cash_opt - w_bond_opt

    # Tính vốn phân bổ theo tối ưu
    capital_cash = w_cash_opt * total_capital
    capital_bond = w_bond_opt * total_capital
    capital_stock = w_stock_opt * total_capital

    capital_alloc = {t: w_stock_opt * total_capital * w for t, w in best_portfolio['Weights'].items()}

    return {
        'capital_cash': capital_cash,
        'capital_bond': capital_bond,
        'capital_stock': capital_stock,
        'capital_alloc': capital_alloc,
        'w_cash': w_cash_opt,
        'w_bond': w_bond_opt,
        'w_stock': w_stock_opt,
        'optimization_success': result.success,
        'optimization_message': result.message
    }


def run(
    hrp_result_dict, adj_returns_combinations, cov_matrix_dict,
    rf, A, total_capital, risk_score,
    alloc_cash, alloc_bond, alloc_stock,
    y_min=0.6, y_max=0.9, time_horizon=None,
    margin=0.03
):
    if not hrp_result_dict:
        raise ValueError("❌ No valid HRP-CVaR portfolios found.")

    best_key = max(hrp_result_dict, key=lambda k: hrp_result_dict[k]['Sharpe Ratio'])
    best_portfolio = hrp_result_dict[best_key]

    tickers = list(best_portfolio['Weights'].keys())

    mu = np.array([adj_returns_combinations[best_key][t] for t in tickers]) / 100
    cov = cov_matrix_dict[best_key].loc[tickers, tickers].values

    target_alloc = {
        'cash': alloc_cash,
        'bond': alloc_bond,
        'stock': alloc_stock
    }

    optimized = optimize_allocation(
        best_portfolio, mu, cov, rf, A, total_capital,
        target_alloc, margin=margin
    )

    capital_cash = optimized['capital_cash']
    capital_bond = optimized['capital_bond']
    capital_stock = optimized['capital_stock']
    capital_alloc = optimized['capital_alloc']

    w_cash = optimized['w_cash']
    w_bond = optimized['w_bond']
    w_stock = optimized['w_stock']

    weights = np.array([best_portfolio['Weights'][t] for t in tickers])
    weights /= weights.sum()

    mu_p = weights @ mu
    sigma_p = np.sqrt(weights.T @ cov @ weights)

    if mu_p <= 0 or sigma_p <= 0:
        raise ValueError("❌ Risky portfolio has invalid return or volatility.")

    expected_rc = (
        capital_stock * mu_p +
        capital_bond * rf +
        capital_cash * rf
    ) / total_capital

    sigma_c = (capital_stock * sigma_p) / total_capital

    utility = expected_rc - 0.5 * A * sigma_c ** 2

    portfolio_info = {
        'portfolio_name': '-'.join(best_key),
        'mu': mu_p,
        'sigma': sigma_p,
        'rf': rf,
        'A': A,
        'risk_score': risk_score,
        'expected_rc': expected_rc,
        'sigma_c': sigma_c,
        'utility': utility,
        'capital_risky': capital_stock,
        'capital_rf': capital_cash + capital_bond,
        'capital_cash': capital_cash,
        'capital_bond': capital_bond,
        'capital_stock': capital_stock,
        'alloc_cash': alloc_cash,
        'alloc_bond': alloc_bond,
        'alloc_stock': alloc_stock,
        'actual_cash_ratio': w_cash,
        'actual_bond_ratio': w_bond,
        'actual_stock_ratio': w_stock,
        'target_cash_ratio': alloc_cash,
        'target_bond_ratio': alloc_bond,
        'target_stock_ratio': alloc_stock,
        'time_horizon': time_horizon,
        'margin': margin,
        'optimization_success': optimized['optimization_success'],
        'optimization_message': optimized['optimization_message']
    }

    return (
        best_portfolio,
        w_stock,
        capital_alloc,
        sigma_c,
        expected_rc,
        weights,
        tickers,
        portfolio_info,
        sigma_p,
        mu,
        mu_p,
        cov
    )
