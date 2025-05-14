# utils/block_g1_bond_model_advanced.py

import numpy as np
from scipy.optimize import newton

def run(bond_price, coupon_rate, face_value, years_to_maturity, r0=0.05, a=0.1, b=0.05, sigma=0.02, n_paths=1000, T=1):
    def bond_price_function(r):
        coupons = sum([
            (coupon_rate * face_value) / (1 + r) ** t
            for t in range(1, years_to_maturity + 1)
        ])
        principal = face_value / (1 + r) ** years_to_maturity
        return coupons + principal

    def estimate_duration(ytm):
        weighted_cashflows = sum([
            (t * coupon_rate * face_value) / (1 + ytm) ** t
            for t in range(1, years_to_maturity + 1)
        ])
        weighted_principal = (years_to_maturity * face_value) / (1 + ytm) ** years_to_maturity
        price = bond_price_function(ytm)
        return (weighted_cashflows + weighted_principal) / price

    # === Estimate YTM ===
    try:
        ytm = newton(lambda r: bond_price_function(r) - bond_price, x0=0.05)
        if not (0 < ytm < 1):
            ytm = 0.05
    except Exception:
        ytm = 0.05

    # === Estimate Volatility using CIR model ===
    np.random.seed(42)
    dt = T / years_to_maturity
    rates = np.zeros((n_paths, years_to_maturity))
    rates[:, 0] = r0

    for t in range(1, years_to_maturity):
        rt = rates[:, t - 1]
        dr = a * (b - rt) * dt + sigma * np.sqrt(np.maximum(rt, 0)) * np.sqrt(dt) * np.random.normal(size=n_paths)
        rates[:, t] = np.maximum(rt + dr, 0)

    final_rates = rates[:, -1]
    bond_volatility = np.std(final_rates)

    return round(ytm, 6), round(bond_volatility, 4), "CUSTOM_BOND"
