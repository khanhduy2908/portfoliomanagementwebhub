
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def run(hrp_cvar_results, adj_returns_combinations, cov_matrix_dict, rf, A):
    # --- Extract Best Portfolio ---
    best_portfolio = max(hrp_cvar_results, key=lambda x: x['Sharpe Ratio'])
    portfolio_name = best_portfolio['Portfolio']
    tickers = list(best_portfolio['Weights'].keys())
    weights = np.array(list(best_portfolio['Weights'].values()))

    # --- Get Returns and Covariance ---
    mu = np.array([adj_returns_combinations[portfolio_name][t] for t in tickers]) / 100
    cov = cov_matrix_dict[portfolio_name].loc[tickers, tickers].values

    def portfolio_performance(w, returns, cov, rf):
        ret = np.dot(w, returns)
        vol = np.sqrt(w.T @ cov @ w)
        sharpe = (ret - rf) / vol if vol > 0 else 0
        return ret, vol, sharpe

    # --- Efficient Frontier ---
    n_portfolios = 5000
    results = np.zeros((3, n_portfolios))
    for i in range(n_portfolios):
        w = np.random.random(len(tickers))
        w /= np.sum(w)
        ret, vol, sr = portfolio_performance(w, mu, cov, rf)
        results[:, i] = [ret, vol, sr]

    # --- Optimal Points ---
    mu_p = mu @ weights
    sigma_p = np.sqrt(weights.T @ cov @ weights)
    y_opt = (mu_p - rf) / (A * sigma_p**2)
    y_capped = max(0.6, min(y_opt, 0.9))
    E_rc = y_capped * mu_p + (1 - y_capped) * rf
    sigma_c = y_capped * sigma_p

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='black')
    ax.set_facecolor('black')

    sc = ax.scatter(results[1, :]*100, results[0, :]*100, c=results[2, :],
                    cmap='viridis', alpha=0.5)
    plt.colorbar(sc, ax=ax, label="Sharpe Ratio")

    ax.scatter(sigma_p*100, mu_p*100, c='red', marker='*', s=200, label='Optimal Risky Portfolio')
    ax.scatter(0, rf*100, c='blue', marker='o', s=100, label=f'Risk-Free Rate ({rf*100:.2f}%)')

    x = np.linspace(0, max(results[1, :])*1.2, 100)
    y = rf + (mu_p - rf) / sigma_p * x
    ax.plot(x*100, y*100, 'r--', label='Capital Allocation Line (CAL)')

    ax.scatter(sigma_c*100, E_rc*100, c='lime', marker='D', s=150,
               label=f'Optimal Complete Portfolio (y={y_capped:.2f})')

    ax.set_title("Efficient Frontier with Capital Allocation Line", color='white')
    ax.set_xlabel("Volatility (%)", color='white')
    ax.set_ylabel("Expected Return (%)", color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor='black', edgecolor='white', labelcolor='white')
    ax.grid(False)

    st.pyplot(fig)
