# utils/block_g_optimization.py

import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import multivariate_normal
import streamlit as st

# --- CONFIG ---
alpha_cvar = 0.95
lambda_cvar = 5          # mức độ phạt CVaR
beta_l2 = 0.01           # regularization
n_simulations = 20000
cvar_soft_limit = 6.5    # giới hạn CVaR mềm (%)
solvers = ['SCS', 'ECOS']

def cov_to_corr(cov):
    std = np.sqrt(np.diag(cov))
    return cov / np.outer(std, std)

def corr_to_dist(corr):
    return np.sqrt(0.5 * (1 - corr))

def run(valid_combinations, adj_returns_combinations, cov_matrix_dict, returns_benchmark):
    hrp_cvar_results = []

    st.markdown("### 🧠 Portfolio Optimization (HRP + CVaR Soft Constraint)")

    benchmark_return_mean = returns_benchmark['Benchmark_Return'].mean()

    for combo in valid_combinations:
        tickers = combo.split('-')
        try:
            mu = np.array([adj_returns_combinations[combo][t] for t in tickers]) / 100
            cov = cov_matrix_dict[combo].loc[tickers, tickers].values

            if np.any(np.linalg.eigvalsh(cov) < -1e-6):
                st.warning(f"❌ `{combo}`: Covariance matrix not PSD.")
                continue

            # HRP clustering order
            corr = cov_to_corr(cov)
            dist = corr_to_dist(corr)
            dist_vec = squareform(dist, checks=False)
            Z = linkage(dist_vec, method='ward')
            clusters = fcluster(Z, t=len(tickers), criterion='maxclust')
            order = np.argsort(clusters)

            mu_ord = mu[order]
            cov_ord = cov[np.ix_(order, order)]
            tickers_ord = [tickers[i] for i in order]

            # Simulate return scenarios
            np.random.seed(42)
            scenarios = multivariate_normal.rvs(mean=mu_ord, cov=cov_ord, size=n_simulations)
            losses = -scenarios

            # CVaR Optimization
            w = cp.Variable(len(tickers))
            VaR = cp.Variable()
            z = cp.Variable(n_simulations)

            port_loss = losses @ w
            cvar = VaR + cp.sum(z) / ((1 - alpha_cvar) * n_simulations)
            objective = cp.Maximize(mu_ord @ w - lambda_cvar * cvar - beta_l2 * cp.sum_squares(w))

            constraints = [
                cp.sum(w) == 1,
                w >= 0,
                z >= 0,
                z >= port_loss - VaR
            ]

            success = False
            for solver in solvers:
                try:
                    prob = cp.Problem(objective, constraints)
                    prob.solve(solver=solver, max_iters=10000)
                    if prob.status in ['optimal', 'optimal_inaccurate']:
                        success = True
                        break
                except Exception:
                    continue

            if not success or w.value is None:
                st.warning(f"❌ `{combo}`: Optimization failed.")
                continue

            w_opt = w.value
            if np.any(np.isnan(w_opt)) or np.abs(np.sum(w_opt) - 1) > 1e-3:
                st.warning(f"⚠️ `{combo}`: Invalid weights.")
                continue

            port_ret = mu_ord @ w_opt * 100
            port_vol = np.sqrt(w_opt.T @ cov_ord @ w_opt) * 100
            final_cvar = (VaR.value + np.mean(np.maximum(losses @ w_opt - VaR.value, 0)) / (1 - alpha_cvar)) * 100
            sharpe = (port_ret - benchmark_return_mean * 100) / port_vol if port_vol > 0 else 0
            exceed_flag = final_cvar > cvar_soft_limit

            hrp_cvar_results.append({
                'Portfolio': combo,
                'Expected Return (%)': port_ret,
                'Volatility (%)': port_vol,
                'CVaR (%)': final_cvar,
                'Sharpe Ratio': sharpe,
                'CVaR Exceed?': exceed_flag,
                'Weights': dict(zip(tickers_ord, w_opt))
            })

        except Exception as e:
            st.warning(f"❌ `{combo}`: {e}")
            continue

    # Reporting
    if hrp_cvar_results:
        hrp_df = pd.DataFrame(hrp_cvar_results).sort_values(by='Sharpe Ratio', ascending=False)
        st.dataframe(hrp_df[['Portfolio', 'Expected Return (%)', 'Volatility (%)',
                             'CVaR (%)', 'Sharpe Ratio', 'CVaR Exceed?']].round(2), use_container_width=True)
    else:
        st.error("❌ No valid portfolios optimized.")

    return hrp_cvar_results
