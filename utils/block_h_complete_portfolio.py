import numpy as np
from scipy.optimize import minimize

def get_max_rf_ratio(score, A, alloc_cash, alloc_bond, alloc_stock):
    if 10 <= score <= 17:
        hard_cap = 0.40
    elif 18 <= score <= 27:
        hard_cap = 0.20
    elif 28 <= score <= 40:
        hard_cap = 0.10
    else:
        raise ValueError("Invalid risk score.")

    if A >= 25:
        suggested = 0.40
    elif A <= 2:
        suggested = 0.02
    else:
        suggested = 0.02 + (A - 2) * ((0.40 - 0.02) / (25 - 2))

    max_target_rf = alloc_cash + alloc_bond + 0.4 * alloc_stock
    return min(hard_cap, suggested, max_target_rf)

def penalty_allocations(w, target_alloc, penalty=1000):
    # w and target_alloc are vectors [cash, bond, stock, risky_fraction_in_stock]
    penalties = 0
    for i in range(len(w)):
        diff = w[i] - target_alloc[i]
        if abs(diff) > 0.05:  # 5% tolerance
            penalties += penalty * diff**2
    return penalties

def utility_function(w, mu_vec, cov_mat, rf, A, target_alloc):
    # w = [cash, bond, stock_total, risky_fraction]
    # Expected return (annualized or monthly as input)
    mu_p = w[0] * rf + w[1] * rf + w[2] * (w[3] * mu_vec.mean() + (1 - w[3]) * rf)
    # Volatility comes only from risky stock portion
    sigma_p = np.sqrt((w[2] * w[3])**2 * np.dot(mu_vec.T, np.dot(cov_mat, mu_vec)))
    utility = mu_p - 0.5 * A * sigma_p**2

    penalty = penalty_allocations(w, target_alloc)
    return -utility + penalty  # We minimize negative utility plus penalty

def run(
    hrp_result_dict, adj_returns_combinations, cov_matrix_dict,
    rf, A, total_capital, risk_score,
    alloc_cash, alloc_bond, alloc_stock,
    y_min=0.6, y_max=0.9, time_horizon=None):

    if not hrp_result_dict:
        raise ValueError("No valid HRP-CVaR portfolios found.")

    best_key = max(hrp_result_dict, key=lambda k: hrp_result_dict[k]['Sharpe Ratio'])
    best_portfolio = hrp_result_dict[best_key]

    tickers = list(best_portfolio['Weights'].keys())
    weights = np.array([best_portfolio['Weights'][t] for t in tickers])
    weights /= weights.sum()

    mu_stock = np.array([adj_returns_combinations[best_key][t] for t in tickers]) / 100
    cov_stock = cov_matrix_dict[best_key].loc[tickers, tickers].values

    # Target allocation vector: cash, bond, stock, risky fraction inside stock
    target_alloc = np.array([alloc_cash, alloc_bond, alloc_stock, 0.7])  # risky_fraction default 70%

    # Initial guess
    w0 = np.array([alloc_cash, alloc_bond, alloc_stock, 0.7])

    bounds = [(max(0, alloc_cash - 0.05), min(1, alloc_cash + 0.05)),
              (max(0, alloc_bond - 0.05), min(1, alloc_bond + 0.05)),
              (max(0, alloc_stock - 0.05), min(1, alloc_stock + 0.05)),
              (y_min, y_max)]

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w[:3]) - 1})  # cash+bond+stock = 1

    res = minimize(
        utility_function,
        w0,
        args=(mu_stock, cov_stock, rf, A, target_alloc),
        bounds=bounds,
        constraints=constraints,
        method='SLSQP',
        options={'ftol': 1e-9, 'disp': False}
    )

    if not res.success:
        raise ValueError(f"Optimization failed: {res.message}")

    w_opt = res.x
    cash_alloc, bond_alloc, stock_alloc, risky_fraction = w_opt

    capital_cash = cash_alloc * total_capital
    capital_bond = bond_alloc * total_capital
    capital_stock = stock_alloc * total_capital
    capital_risky = capital_stock * risky_fraction
    capital_rf_total = capital_cash + capital_bond + capital_stock * (1 - risky_fraction)

    capital_alloc = {t: capital_risky * w for t, w in zip(tickers, weights)}

    expected_rc = (
        capital_cash * rf + capital_bond * rf +
        capital_stock * (risky_fraction * mu_stock.mean() + (1 - risky_fraction) * rf)
    ) / total_capital

    sigma_c = (capital_stock * risky_fraction * np.sqrt(np.dot(weights.T, np.dot(cov_stock, weights)))) / total_capital

    utility = expected_rc - 0.5 * A * sigma_c ** 2

    max_rf_ratio = get_max_rf_ratio(risk_score, A, alloc_cash, alloc_bond, alloc_stock)

    portfolio_info = {
        'portfolio_name': '-'.join(best_key),
        'mu': mu_stock.mean(),
        'sigma': np.sqrt(np.dot(weights.T, np.dot(cov_stock, weights))),
        'rf': rf,
        'A': A,
        'risk_score': risk_score,
        'y_opt': risky_fraction,
        'y_capped': risky_fraction,
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
        'actual_cash_ratio': cash_alloc,
        'actual_bond_ratio': bond_alloc,
        'actual_stock_ratio': stock_alloc,
        'target_cash_ratio': alloc_cash,
        'target_bond_ratio': alloc_bond,
        'target_stock_ratio': alloc_stock,
        'max_rf_ratio': max_rf_ratio,
        'time_horizon': time_horizon
    }

    return (
        best_portfolio, risky_fraction, capital_alloc,
        sigma_c, expected_rc, weights, tickers,
        portfolio_info, np.sqrt(np.dot(weights.T, np.dot(cov_stock, weights))),
        mu_stock, risky_fraction, mu_stock.mean(), cov_stock
    )
