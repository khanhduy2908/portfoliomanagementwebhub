import numpy as np
from scipy.optimize import minimize

def optimize_y_opt(mu_p, sigma_p, rf, A, y_min, y_max):
    """
    Tối ưu tỉ lệ rủi ro y (risk exposure) trong khoảng [y_min, y_max]
    maximize utility = E[r] - 0.5 * A * var[r]
    """
    def neg_utility(y):
        expected_return = y * mu_p + (1 - y) * rf
        volatility = y * sigma_p
        return -(expected_return - 0.5 * A * volatility ** 2)

    # Chạy tối ưu bounded với method 'bounded'
    res = minimize(neg_utility, x0=(y_min + y_max) / 2, bounds=[(y_min, y_max)], method='bounded')
    if not res.success:
        raise RuntimeError(f"Optimization y_opt failed: {res.message}")
    return float(res.x)

def optimize_allocation(
    best_portfolio, mu, cov, rf, A, total_capital,
    target_alloc, margin=0.03
):
    """
    Tối ưu phân bổ vốn cash, bond, stock thỏa target ± margin (penalty mềm)
    """

    weights_stock_i = np.array([best_portfolio['Weights'][t] for t in best_portfolio['Weights']])
    if np.sum(weights_stock_i) <= 0:
        raise ValueError("Invalid weights in best_portfolio.")
    weights_stock_i /= np.sum(weights_stock_i)

    def smooth_penalty(x, target, margin):
        diff = abs(x - target)
        return 0 if diff <= margin else 1000 * (diff - margin) ** 2

    def utility(x):
        w_cash, w_bond = x
        w_stock = 1 - w_cash - w_bond

        # Kiểm tra giới hạn cơ bản tránh out-of-bound
        if any([
            w_cash < 0, w_bond < 0, w_stock < 0,
            w_cash > 1, w_bond > 1, w_stock > 1
        ]):
            return 1e8

        expected_return = w_stock * np.dot(weights_stock_i, mu) + (w_bond + w_cash) * rf
        volatility = np.sqrt(weights_stock_i.T @ cov @ weights_stock_i) * w_stock
        u = expected_return - 0.5 * A * volatility ** 2

        penalty = (
            smooth_penalty(w_cash, target_alloc['cash'], margin) +
            smooth_penalty(w_bond, target_alloc['bond'], margin) +
            smooth_penalty(w_stock, target_alloc['stock'], margin)
        )

        # Trả về hàm mục tiêu (âm utility + penalty)
        return -u + penalty

    constraints = ({
        'type': 'eq',
        'fun': lambda x: 1 - sum(x) - (1 - sum(x))
    })

    bounds = [(0, 1), (0, 1)]

    initial_guess = [
        np.clip(target_alloc['cash'], margin, 1 - 2*margin),
        np.clip(target_alloc['bond'], margin, 1 - 2*margin)
    ]

    result = minimize(utility, initial_guess, bounds=bounds, constraints=constraints, method='SLSQP', options={'ftol':1e-9, 'disp': False})

    if not result.success:
        # fallback tối ưu trust-constr nếu SLSQP fail
        result = minimize(utility, initial_guess, bounds=bounds, constraints=constraints, method='trust-constr', options={'xtol':1e-9, 'disp': False})
        if not result.success:
            raise RuntimeError(f"Optimization allocation failed: {result.message}")

    w_cash_opt, w_bond_opt = result.x
    w_stock_opt = 1 - w_cash_opt - w_bond_opt

    # Ràng buộc lại tổng 1 và các tỉ lệ nằm trong [0,1]
    w_cash_opt = np.clip(w_cash_opt, 0, 1)
    w_bond_opt = np.clip(w_bond_opt, 0, 1)
    w_stock_opt = np.clip(w_stock_opt, 0, 1)

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
    Block H: Tối ưu danh mục hoàn chỉnh với ràng buộc allocation mềm ±margin,
    ràng buộc tỷ trọng phù hợp với chiến lược rủi ro khách hàng,
    tối ưu y và phân bổ cash-bond-stock riêng biệt,
    kiểm soát lỗi chặt chẽ, hiệu quả, đảm bảo không lỗi solver.
    """

    if not hrp_result_dict:
        raise ValueError("❌ No valid HRP-CVaR portfolios found.")

    best_key = max(hrp_result_dict, key=lambda k: hrp_result_dict[k].get('Sharpe Ratio', -np.inf))
    best_portfolio = hrp_result_dict[best_key]

    tickers = list(best_portfolio['Weights'].keys())

    mu_dict = adj_returns_combinations.get(best_key)
    cov_df = cov_matrix_dict.get(best_key)

    if mu_dict is None or cov_df is None:
        raise ValueError("Missing return or covariance data for best portfolio key.")

    mu = np.array([mu_dict[t] for t in tickers]) / 100
    cov = cov_df.loc[tickers, tickers].values

    target_alloc = {
        'cash': alloc_cash,
        'bond': alloc_bond,
        'stock': alloc_stock
    }

    weights = np.array([best_portfolio['Weights'][t] for t in tickers])
    weights /= weights.sum()

    mu_p = weights @ mu
    sigma_p = np.sqrt(weights.T @ cov @ weights)

    if mu_p <= 0 or sigma_p <= 0:
        raise ValueError("❌ Risky portfolio has invalid return or volatility.")

    # Optimize y first
    y_opt = optimize_y_opt(mu_p, sigma_p, rf, A, y_min, y_max)

    # Then optimize allocation given y_opt and constraints
    w_cash, w_bond, w_stock, capital_cash, capital_bond, capital_stock, capital_alloc = optimize_allocation(
        best_portfolio, mu, cov, rf, A, total_capital, target_alloc, margin=margin
    )

    # Calculate expected return and volatility for the complete portfolio
    expected_rc = (
        capital_stock * (mu_p * y_opt + (1 - y_opt) * rf) +
        capital_bond * rf +
        capital_cash * rf
    ) / total_capital

    sigma_c = (capital_stock * y_opt * sigma_p) / total_capital

    utility = expected_rc - 0.5 * A * sigma_c ** 2

    portfolio_info = {
        'portfolio_name': '-'.join(best_key),
        'mu': mu_p,
        'sigma': sigma_p,
        'rf': rf,
        'A': A,
        'risk_score': risk_score,
        'y_opt': y_opt,
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
        'margin': margin
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
        y_opt,
        mu_p,
        cov
    )
