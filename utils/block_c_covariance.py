def run(selected_tickers, returns_pivot_stocks):
    import numpy as np
    import pandas as pd
    from arch import arch_model
    from sklearn.covariance import LedoitWolf

    def compute_garch_volatility(series):
        try:
            model = arch_model(series.dropna(), vol='Garch', p=1, q=1)
            result = model.fit(disp='off')
            return result.conditional_volatility
        except:
            return pd.Series(index=series.index, data=np.nan)

    returns = returns_pivot_stocks[selected_tickers].dropna()
    garch_vols = pd.DataFrame(index=returns.index, columns=selected_tickers)
    for ticker in selected_tickers:
        garch_vols[ticker] = compute_garch_volatility(returns[ticker])

    garch_vols = garch_vols.ffill().bfill()
    std_vector = garch_vols.iloc[-1] / 100

    corr_matrix = returns.corr().values
    D = np.diag(std_vector.values)
    cov_garch = D @ corr_matrix @ D

    lw = LedoitWolf()
    cov_lw = lw.fit(returns).covariance_

    weight_garch = 0.6
    cov_combined = weight_garch * cov_garch + (1 - weight_garch) * cov_lw

    cov_matrix = pd.DataFrame(cov_combined, index=selected_tickers, columns=selected_tickers)
    return cov_matrix
