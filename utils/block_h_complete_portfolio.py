import numpy as np
import cvxpy as cp

def optimize_allocation_cvxpy(
    best_portfolio, mu, cov, rf, A, total_capital,
    target_alloc, margin=0.03
):
    """
    Tối ưu phân bổ tài sản (cash, bond, stock) dùng cvxpy,
    với ràng buộc allocation nằm trong target ± margin,
    tối ưu utility = expected return - 0.5 * A * variance.
    """

    weights_stock_i = np.array([best_portfolio['Weights'][t] for t in best_portfolio['Weights']])
    weights_stock_i /= weights_stock_i.sum()

    # Biến quyết định
    w_cash = cp.Variable()
    w_bond = cp.Variable()
    w_stock = cp.Variable()

    # Tính kỳ vọng lợi nhuận và rủi ro
    expected_return = w_stock * mu @ weights_stock_i + (w_bond + w_cash) * rf
    volatility = cp.sqrt(cp.quad_form(weights_stock_i * w_stock, cov))

    # Hàm mục tiêu utility
    utility = expected_return - 0.5 * A * volatility ** 2

    # Ràng buộc
    constraints = [
        w_cash + w_bond + w_stock == 1,
        w_cash >= max(0, target_alloc['cash'] - margin),
        w_cash <= min(1, target_alloc['cash'] + margin),
        w_bond >= max(0, target_alloc['bond'] - margin),
        w_bond <= min(1, target_alloc['bond'] + margin),
        w_stock >= max(0, target_alloc['stock'] - margin),
        w_stock <= min(1, target_alloc['stock'] + margin),
    ]

    # Định nghĩa và giải bài toán tối ưu
    prob = cp.Problem(cp.Maximize(utility), constraints)

    prob.solve(solver=cp.SCS, verbose=False)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"Optimization failed with status: {prob.status}")

    w_cash_opt = w_cash.value
    w_bond_opt = w_bond.value
    w_stock_opt = w_stock.value

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
    Block H: Tối ưu phân bổ hoàn chỉnh dùng cvxpy,
    chặt chẽ với ràng buộc allocation ± margin,
    đảm bảo hiệu quả và tính ổn định.
    """

    if not hrp_result_dict:
        raise ValueError("❌ No valid HRP-CVaR portfolios found.")

    # Chọn portfolio có Sharpe Ratio cao nhất
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

    # Tối ưu allocation
    w_cash, w_bond, w_stock, capital_cash, capital_bond, capital_stock, capital_alloc = optimize_allocation_cvxpy(
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
        mu_p,
        cov,
        w_cash
    )
