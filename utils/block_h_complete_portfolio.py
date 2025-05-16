import numpy as np
from scipy.optimize import minimize

def optimize_allocation(
    best_portfolio, mu, cov, rf, A, total_capital,
    target_alloc, max_deviation=0.03, penalty_weight=10000
):
    """
    Tối ưu phân bổ tài sản với penalty mềm đảm bảo sai lệch với mục tiêu nhỏ hơn ±3%.
    """
    weights_stock_i = np.array([best_portfolio['Weights'][t] for t in best_portfolio['Weights']])
    weights_stock_i /= weights_stock_i.sum()

    def penalty(x):
        diff = np.abs(x - np.array([target_alloc['cash'], target_alloc['bond'], target_alloc['stock']]))
        penalty_vec = np.where(diff > max_deviation, penalty_weight * (diff - max_deviation) ** 2, 0)
        return penalty_vec.sum()

    def objective(x):
        w_cash, w_bond = x
        w_stock = 1 - w_cash - w_bond
        if min(w_cash, w_bond, w_stock) < 0 or max(w_cash, w_bond, w_stock) > 1:
            return 1e6
        ret = w_stock * np.dot(weights_stock_i, mu) + (w_cash + w_bond) * rf
        vol = np.sqrt(weights_stock_i.T @ cov @ weights_stock_i) * w_stock
        util = ret - 0.5 * A * vol ** 2
        return -util + penalty(np.array([w_cash, w_bond, w_stock]))

    initial_guess = [
        np.clip(target_alloc['cash'], max_deviation, 1 - 2 * max_deviation),
        np.clip(target_alloc['bond'], max_deviation, 1 - 2 * max_deviation)
    ]
    constraints = [{'type': 'eq', 'fun': lambda x: 1 - (x[0] + x[1] + (1 - x[0] - x[1]))}]
    bounds = [(0, 1), (0, 1)]

    result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints, method='SLSQP')
    if not result.success:
        raise ValueError("⚠️ Optimization failed:", result.message)

    w_cash, w_bond = result.x
    w_stock = 1 - w_cash - w_bond

    capital_cash = w_cash * total_capital
    capital_bond = w_bond * total_capital
    capital_stock = w_stock * total_capital
    capital_alloc = {t: capital_stock * w for t, w in best_portfolio['Weights'].items()}

    return w_cash, w_bond, w_stock, capital_cash, capital_bond, capital_stock, capital_alloc

def run(
    hrp_result_dict, adj_returns_combinations, cov_matrix_dict,
    rf, A, total_capital, risk_score,
    alloc_cash, alloc_bond, alloc_stock,
    y_min=0.6, y_max=0.9, time_horizon=None
):
    if not hrp_result_dict:
        raise ValueError("❌ No valid HRP-CVaR portfolios found.")

    best_key = max(hrp_result_dict, key=lambda k: hrp_result_dict[k]['Sharpe Ratio'])
    best_portfolio = hrp_result_dict[best_key]
    tickers = list(best_portfolio['Weights'].keys())

    mu = np.array([adj_returns_combinations[best_key][t] for t in tickers]) / 100
    cov = cov_matrix_dict[best_key].loc[tickers, tickers].values

    target_alloc = {"cash": alloc_cash, "bond": alloc_bond, "stock": alloc_stock}

    w_cash, w_bond, w_stock, capital_cash, capital_bond, capital_stock, capital_alloc = optimize_allocation(
        best_portfolio, mu, cov, rf, A, total_capital, target_alloc
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

    y_opt = w_stock
    y_capped = min(max(y_min, y_opt), y_max)

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
        'y_opt': y_opt,
        'y_capped': y_capped,
        'time_horizon': time_horizon
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
        w_cash,
        y_opt,
        y_capped
    )
