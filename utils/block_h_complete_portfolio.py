import numpy as np
from scipy.optimize import minimize

def optimize_allocation(
    best_portfolio, mu, cov, rf, A, total_capital,
    target_alloc, margin=0.03
):
    """
    Tối ưu tỷ trọng phân bổ (cash, bond, stock) thỏa ràng buộc target ± margin dưới dạng soft penalty.
    Tránh lỗi ràng buộc cứng bằng phương pháp penalty hàm mục tiêu.
    """

    weights_stock_i = np.array([best_portfolio['Weights'][t] for t in best_portfolio['Weights']])
    weights_stock_i /= weights_stock_i.sum()

    def smooth_penalty(x, target, margin):
        diff = abs(x - target)
        if diff <= margin:
            return 0
        else:
            # penalty quadratic bắt đầu từ margin, số lớn để ưu tiên ràng buộc
            return 1000 * (diff - margin) ** 2

    def utility(x):
        w_cash, w_bond = x
        w_stock = 1 - w_cash - w_bond

        # Giới hạn cơ bản để tránh out-of-bound
        if (w_stock < 0) or (w_cash < 0) or (w_bond < 0) or (w_cash > 1) or (w_bond > 1):
            return 1e8

        expected_return = w_stock * np.dot(weights_stock_i, mu) + (w_bond + w_cash) * rf
        volatility = np.sqrt(weights_stock_i.T @ cov @ weights_stock_i) * w_stock
        u = expected_return - 0.5 * A * volatility ** 2

        # Phạt lệch allocation vượt margin
        penalty = (
            smooth_penalty(w_cash, target_alloc['cash'], margin) +
            smooth_penalty(w_bond, target_alloc['bond'], margin) +
            smooth_penalty(w_stock, target_alloc['stock'], margin)
        )

        # Mục tiêu tối ưu: maximize utility - penalty → minimize negative utility + penalty
        return -u + penalty

    # Ràng buộc tổng tỷ trọng = 1
    constraints = ({
        'type': 'eq',
        'fun': lambda x: 1 - (x[0] + x[1] + (1 - x[0] - x[1]))
    })

    # Bounds cho cash và bond trong [0,1]
    bounds = [(0,1), (0,1)]

    # Điểm khởi đầu gần target allocation trong bounds hợp lý
    initial_guess = [
        np.clip(target_alloc['cash'], margin, 1 - 2*margin),
        np.clip(target_alloc['bond'], margin, 1 - 2*margin)
    ]

    # Tối ưu bằng SLSQP, fallback trust-constr nếu không thành công
    result = minimize(utility, x0=initial_guess, bounds=bounds, constraints=constraints, method='SLSQP', options={'ftol':1e-9, 'disp': False})

    if not result.success:
        result = minimize(utility, x0=initial_guess, bounds=bounds, constraints=constraints, method='trust-constr', options={'xtol':1e-9, 'disp': False})
        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")

    w_cash_opt, w_bond_opt = result.x
    w_stock_opt = 1 - w_cash_opt - w_bond_opt

    capital_cash = w_cash_opt * total_capital
    capital_bond = w_bond_opt * total_capital
    capital_stock = w_stock_opt * total_capital

    capital_alloc = {t: w_stock_opt * total_capital * w for t, w in best_portfolio['Weights'].items()}

    return w_cash_opt, w_bond_opt, w_stock_opt, capital_cash, capital_bond, capital_stock, capital_alloc


def run(
    hrp_result_dict, adj_returns_combinations, cov_matrix_dict,
    rf, A, total_capital, risk_score,
    alloc_cash, alloc_bond, alloc_stock,
    y_min=0.6, y_max=0.9, time_horizon=None,
    margin=0.03
):
    """
    Block H: Tối ưu hoàn chỉnh danh mục với ràng buộc allocation mềm ±margin,
    đảm bảo tỷ trọng phù hợp với chiến lược rủi ro khách hàng,
    tránh lỗi tối ưu và đảm bảo tính nhất quán.
    """

    if not hrp_result_dict:
        raise ValueError("❌ No valid HRP-CVaR portfolios found.")

    # Chọn portfolio tốt nhất theo Sharpe Ratio
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

    # Gọi hàm tối ưu allocation
    w_cash, w_bond, w_stock, capital_cash, capital_bond, capital_stock, capital_alloc = optimize_allocation(
        best_portfolio, mu, cov, rf, A, total_capital,
        target_alloc, margin=margin
    )

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

    # Tính max_rf_ratio (giới hạn tối đa vốn không rủi ro)
    max_rf_ratio = alloc_cash + alloc_bond

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
        'y_opt': w_stock,     # Trả về y_opt để block h1 dùng
        'y_capped': w_stock,  # Trả về y_capped (bằng y_opt ở đây)
        'max_rf_ratio': max_rf_ratio
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
        cov,
        w_cash
    )
