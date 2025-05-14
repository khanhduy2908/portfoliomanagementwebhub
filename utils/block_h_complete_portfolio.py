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


def run(
    hrp_result_dict, adj_returns_combinations, cov_matrix_dict,
    rf, A, total_capital, risk_score,
    alloc_cash, alloc_bond, alloc_stock,
    y_min=0.6, y_max=0.9, time_horizon=None):

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

    capital_stock = alloc_stock * total_capital
    capital_cash_init = alloc_cash * total_capital
    capital_bond_init = alloc_bond * total_capital

    max_rf_ratio = get_max_rf_ratio(risk_score, A, alloc_cash, alloc_bond, alloc_stock)

    target_alloc = np.array([alloc_cash, alloc_bond, alloc_stock])

    def loss_fn(y):
        capital_risky = capital_stock * y
        capital_rf_internal = capital_stock * (1 - y)

        capital_cash = capital_cash_init
        capital_bond = capital_bond_init

        rf_internal_split = capital_rf_internal * np.array([
            alloc_cash / (alloc_cash + alloc_bond),
            alloc_bond / (alloc_cash + alloc_bond)
        ])

        capital_cash += rf_internal_split[0]
        capital_bond += rf_internal_split[1]

        capital_rf_total = capital_cash + capital_bond
        actual_alloc = np.array([
            capital_cash / total_capital,
            capital_bond / total_capital,
            capital_risky / total_capital
        ])

        expected_return = y * mu_p + (1 - y) * rf
        volatility = y * sigma_p
        utility = expected_return - 0.5 * A * volatility**2

        alloc_penalty = np.sum((actual_alloc - target_alloc)**2)
        rf_penalty = max(0, capital_rf_total / total_capital - max_rf_ratio)

        λ1, λ2 = 10.0, 100.0
        return -utility + λ1 * alloc_penalty + λ2 * rf_penalty

    res = minimize(loss_fn, x0=[(y_min + y_max) / 2], bounds=[(y_min, y_max)])
    y_opt = float(res.x[0])
    y_capped = np.clip(y_opt, y_min, y_max)

    capital_risky = capital_stock * y_capped
    capital_rf_internal = capital_stock * (1 - y_capped)

    rf_internal_split = capital_rf_internal * np.array([
        alloc_cash / (alloc_cash + alloc_bond),
        alloc_bond / (alloc_cash + alloc_bond)
    ])

    capital_cash = capital_cash_init + rf_internal_split[0]
    capital_bond = capital_bond_init + rf_internal_split[1]
    capital_rf_total = capital_cash + capital_bond

    capital_alloc = {t: capital_risky * w for t, w in zip(tickers, weights)}

    expected_rc = (
        capital_stock * (y_capped * mu_p + (1 - y_capped) * rf) +
        capital_bond * rf +
        capital_cash * rf
    ) / total_capital

    sigma_c = (capital_stock * y_capped * sigma_p) / total_capital
    utility = expected_rc - 0.5 * A * sigma_c**2

    actual_cash_ratio = capital_cash / total_capital
    actual_bond_ratio = capital_bond / total_capital
    actual_stock_ratio = capital_risky / total_capital

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
        'max_rf_ratio': max_rf_ratio,
        'time_horizon': time_horizon
    }

    return (
        best_portfolio, y_capped, capital_alloc,
        sigma_c, expected_rc, weights, tickers,
        portfolio_info, sigma_p, mu, y_opt, mu_p, cov
    )
