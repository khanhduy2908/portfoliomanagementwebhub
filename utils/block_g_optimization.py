import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import multivariate_normal
from numpy.linalg import LinAlgError

import config

def cov_to_corr(cov):
    std = np.sqrt(np.diag(cov))
    return cov / np.outer(std, std)

def corr_to_dist(corr):
    return np.sqrt(0.5 * (1 - corr))

def run(valid_combinations, adj_returns_combinations, cov_matrix_dict, returns_benchmark):
    print("📌 Block G: Portfolio Optimization (HRP + Soft CVaR)...")

    benchmark_return_mean = returns_benchmark['Benchmark_Return'].mean()
    hrp_cvar_results = []

    for combo in valid_combinations:
        tickers = combo.split('-')
        try:
            mu = np.array([adj_returns_combinations[combo][t] for t in tickers]) / 100
            cov = cov_matrix_dict[combo].loc[tickers, tickers].values

            # --- PSD Check ---
            if np.any(np.linalg.eigvalsh(cov) < -1e-6):
                print(f"{combo}: Covariance matrix not PSD.")
                continue

            # --- HRP Ordering ---
            corr = cov_to_corr(cov)
            dist = corr_to_dist(corr)
            dist_vec = squareform(dist, checks=False)
            Z = linkage(dist_vec, method='ward')
            clusters = fcluster(Z, t=len(tickers), criterion='maxclust')
            order = np.argsort(clusters)

            mu_ord = mu[order]
            cov_ord = cov[np.ix_(order, order)]
            tickers_ord = [tickers[i] for i in order]

            # --- Simulate Return Scenarios ---
            np.random.seed(config.SEED)
            scenarios = multivariate_normal.rvs(mean=mu_ord, cov=cov_ord, size=config.N_SIMULATIONS)
            losses = -scenarios

            # --- Optimization Variables ---
            w = cp.Variable(len(tickers))
            VaR = cp.Variable()
            z = cp.Variable(config.N_SIMULATIONS)

            port_loss = losses @ w
            cvar = VaR + cp.sum(z) / ((1 - config.CVaR_ALPHA) * config.N_SIMULATIONS)

            objective = cp.Maximize(mu_ord @ w - config.LAMBDA_CVaR * cvar - config.BETA_L2 * cp.sum_squares(w))
            constraints = [
                cp.sum(w) == 1,
                w >= 0,
                z >= 0,
                z >= port_loss - VaR
            ]

            success = False
            for solver in config.SOLVERS:
                try:
                    prob = cp.Problem(objective, constraints)
                    prob.solve(solver=solver, max_iters=10000)
                    if prob.status in ['optimal', 'optimal_inaccurate']:
                        success = True
                        break
                except Exception:
                    continue

            if not success or w.value is None:
                print(f"❌ {combo}: Optimization failed.")
                continue

            w_opt = w.value
            if np.any(np.isnan(w_opt)) or np.abs(np.sum(w_opt) - 1) > 1e-3:
                print(f"{combo}: Invalid weights.")
                continue

            # --- Metrics ---
            port_ret = mu_ord @ w_opt * 100
            port_vol = np.sqrt(w_opt.T @ cov_ord @ w_opt) * 100
            final_cvar = (VaR.value + np.mean(np.maximum(losses @ w_opt - VaR.value, 0)) / (1 - config.CVaR_ALPHA)) * 100
            sharpe = (port_ret - benchmark_return_mean * 100) / port_vol if port_vol > 0 else 0
            exceed_flag = final_cvar > config.CVaR_SOFT_LIMIT

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
            print(f"{combo}: {e}")
            continue

    return hrp_cvar_results
