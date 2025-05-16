import numpy as np
from scipy.optimize import minimize

def optimize_allocation(
    best_portfolio, mu, cov, rf, A, total_capital,
    target_alloc, margin=0.03, penalty_weight=5000
):
    weights_stock_i = np.array([best_portfolio['Weights'][t] for t in best_portfolio['Weights']])
    weights_stock_i /= weights_stock_i.sum()

    def smooth_penalty(x, target):
        return penalty_weight * np.sum(np.abs(x - target) ** 1.5)

    def utility(x):
        w_cash, w_bond = x
        w_stock = 1 - w_cash - w_bond

        if (w_stock < 0) or (w_cash < 0) or (w_bond < 0):
            return 1e8

        expected_return = w_stock * np.dot(weights_stock_i, mu) + (w_bond + w_cash) * rf
        volatility = np.sqrt(weights_stock_i.T @ cov @ weights_stock_i) * w_stock
        u = expected_return - 0.5 * A * volatility ** 2

        penalty = smooth_penalty(np.array([w_cash, w_bond, w_stock]),
                                 np.array([target_alloc['cash'], target_alloc['bond'], target_alloc['stock']]))

        return -u + penalty

    constraints = ({
        'type': 'eq',
        'fun': lambda x: 1 - (x[0] + x[1] + (1 - x[0] - x[1]))
    })

    bounds = [(0, 1), (0, 1)]
    initial_guess = [target_alloc['cash'], target_alloc['bond']]

    result = minimize(utility, x0=initial_guess, bounds=bounds, constraints=constraints,
                      method='trust-constr', options={'xtol': 1e-9, 'disp': False})

    if not result.success:
        result = minimize(utility, x0=initial_guess, bounds=bounds, constraints=constraints,
                          method='SLSQP', options={'ftol': 1e-9, 'disp': False})
        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")

    w_cash_opt, w_bond_opt = result.x
    w_stock_opt = 1 - w_cash_opt - w_bond_opt

    capital_cash = w_cash_opt * total_capital
    capital_bond = w_bond_opt * total_capital
    capital_stock = w_stock_opt * total_capital

    capital_alloc = {
        t: w_stock_opt * total_capital * w
        for t, w in best_portfolio['Weights'].items()
    }

    return w_cash_opt, w_bond_opt, w_stock_opt, capital_cash, capital_bond, capital_stock, capital_alloc


def get_max_rf_ratio(risk_score, A, alloc_cash, alloc_bond, alloc_stock):
    if 10 <= risk_score <= 17:
        hard_cap = 0.40
    elif 18 <= risk_score <= 27:
        hard_cap = 0.20
    elif 28 <= risk_score <= 40:
        hard_cap = 0.10
    else:
        hard_cap = 0.40

    suggested = 0.02 + (max(2, min(A, 25)) - 2) * ((0.40 - 0.02) / (25 - 2))
    max_target_rf = alloc_cash + alloc_bond + 0.4 * alloc_stock

    return min(hard_cap, suggested, max_target_rf)


def run(
    hrp_result_dict, adj_returns_combinations, cov_matrix_dict,
    rf, A, total_capital, risk_score,
    alloc_cash, alloc_bond, alloc_stock,
    y_min=0.6, y_max=0.9, time_horizon=None,
    margin=0.03, penalty_weight=5000
):
    if not hrp_result_dict:
        raise ValueError("❌ No valid HRP-CVaR portfolios found.")

    best_key = max(hrp_result_dict, key=lambda k: hrp_result_dict[k]['Sharpe Ratio'])
    best_portfolio = hrp_result_dict[best_key]
    tickers = list(best_portfolio['Weights'].keys())

    mu = np.array([adj_returns_combinations[best_key][t] for t in tickers]) / 100
    cov = cov_matrix_dict[best_key].loc[tickers, tickers].values

    target_alloc = {'cash': alloc_cash, 'bond': alloc_bond, 'stock': alloc_stock}

    w_cash, w_bond, w_stock, capital_cash, capital_bond, capital_stock, capital_alloc = optimize_allocation(
        best_portfolio, mu, cov, rf, A, total_capital, target_alloc, margin=margin, penalty_weight=penalty_weight
    )

    weights = np.array([best_portfolio['Weights'][t] for t in tickers])
    weights /= weights.sum()
    mu_p = weights @ mu
    sigma_p = np.sqrt(weights.T @ cov @ weights)

    if mu_p <= 0 or sigma_p <= 0:
        raise ValueError("❌ Risky portfolio has invalid return or volatility.")

    expected_rc = (capital_stock * mu_p + capital_bond * rf + capital_cash * rf) / total_capital
    sigma_c = (capital_stock * sigma_p) / total_capital
    utility = expected_rc - 0.5 * A * sigma_c ** 2

    max_rf_ratio = get_max_rf_ratio(risk_score, A, alloc_cash, alloc_bond, alloc_stock)

    capital_rf_total = capital_cash + capital_bond
    if capital_rf_total / total_capital > max_rf_ratio:
        excess = capital_rf_total - max_rf_ratio * total_capital
        factor = alloc_cash / (alloc_cash + alloc_bond)
        capital_cash = max(capital_cash - excess * factor, 0)
        capital_bond = max(capital_bond - excess * (1 - factor), 0)
        capital_stock = total_capital - capital_cash - capital_bond
        w_cash = capital_cash / total_capital
        w_bond = capital_bond / total_capital
        w_stock = capital_stock / total_capital

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
        'max_rf_ratio': max_rf_ratio,
        'y_opt': y_opt,
        'y_capped': y_capped,
        'time_horizon': time_horizon,
        'margin': margin,
        'penalty_weight': penalty_weight
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
