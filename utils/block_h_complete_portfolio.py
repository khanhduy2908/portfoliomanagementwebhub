def run(hrp_df, total_capital, rf_monthly, A):
    import numpy as np
    import pandas as pd

    if hrp_df.empty:
        return None, None, None, None, None, None

    # Chọn danh mục tốt nhất dựa trên Sharpe
    best_portfolio = hrp_df.iloc[0]
    weights = best_portfolio['Weights']
    mu = best_portfolio['Expected Return (%)'] / 100
    sigma = best_portfolio['Volatility (%)'] / 100

    # 1. Tính tỷ trọng đầu tư vào danh mục rủi ro (y*)
    y_opt = (mu - rf_monthly) / (A * sigma**2)
    y_capped = max(0, min(y_opt, 1))

    # 2. Hàm tiện ích
    utility = mu - 0.5 * A * sigma**2

    # 3. Phân bổ vốn
    capital_risky = y_capped * total_capital
    capital_rf = total_capital - capital_risky

    capital_stocks = {stock: round(w * capital_risky) for stock, w in weights.items()}
    capital_rf = round(capital_rf)

    return {
        'best_portfolio': best_portfolio['Portfolio'],
        'weights': weights,
        'expected_return': round(mu * 100, 2),
        'volatility': round(sigma * 100, 2),
        'sharpe': round(best_portfolio['Sharpe Ratio'], 3),
        'cvar': round(best_portfolio['CVaR (%)'], 2),
        'capital_allocation': {
            'Risk-Free': capital_rf,
            **capital_stocks
        },
        'capital_risky': capital_risky,
        'capital_rf': capital_rf,
        'y_opt': y_opt,
        'y_capped': y_capped,
        'utility': round(utility, 4)
    }
