# utils/block_g_optimization.py

import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import multivariate_normal
import warnings

def cov_to_corr(cov):
    std = np.sqrt(np.diag(cov))
    return cov / np.outer(std, std)

def corr_to_dist(corr):
    return np.sqrt(0.5 * (1 - corr))

def run(valid_combinations, adj_returns_combinations, cov_matrix_dict, returns_benchmark,
        alpha_cvar=0.95, lambda_cvar=5, beta_l2=0.01, cvar_soft_limit=6.5, n_simulations=15000):

    benchmark_return_mean = returns_benchmark['Benchmark_Return'].mean()
    real_portfolios = []
    fallback_result = None
    solvers = ['SCS', 'ECOS']

    # === 1. Danh mục tối ưu hóa thực tế ===
    for combo in valid_combinations:
        tickers = list(combo)
        try:
            mu_dict = adj_returns_combinations[combo]
            cov_df = cov_matrix_dict[combo]

            mu = np.array([mu_dict[t] for t in tickers]) / 100
            cov = cov_df.loc[tickers, tickers].values
            if np.any(np.linalg.eigvalsh(cov) < -1e-6):
                continue

            corr = cov_to_corr(cov)
            dist = corr_to_dist(corr)
            Z = linkage(squareform(dist, checks=False), method='ward')

            for seed in range(3):  # Multiple clustering seeds
                np.random.seed(seed)
                clusters = fcluster(Z, t=len(tickers), criterion='maxclust')
                order = np.argsort(clusters + np.random.rand(len(clusters)) * 0.01)

                mu_ord = mu[order]
                cov_ord = cov[np.ix_(order, order)]
                tickers_ord = [tickers[i] for i in order]

                scenarios = multivariate_normal.rvs(mean=mu_ord, cov=cov_ord, size=n_simulations)
                losses = -scenarios

                w = cp.Variable(len(tickers))
                VaR = cp.Variable()
                z = cp.Variable(n_simulations)

                port_loss = losses @ w
                cvar = VaR + cp.sum(z) / ((1 - alpha_cvar) * n_simulations)
                objective = cp.Maximize(mu_ord @ w - lambda_cvar * cvar - beta_l2 * cp.sum_squares(w))
                constraints = [cp.sum(w) == 1, w >= 0, z >= 0, z >= port_loss - VaR]

                success = False
                for solver in solvers:
                    try:
                        prob = cp.Problem(objective, constraints)
                        prob.solve(solver=solver, max_iters=5000)
                        if prob.status in ['optimal', 'optimal_inaccurate']:
                            success = True
                            break
                    except:
                        continue

                if not success or w.value is None:
                    continue

                w_opt = w.value
                if np.any(np.isnan(w_opt)) or abs(np.sum(w_opt) - 1) > 1e-3:
                    continue

                port_ret = mu_ord @ w_opt * 100
                port_vol = np.sqrt(w_opt.T @ cov_ord @ w_opt) * 100
                final_cvar = (VaR.value + np.mean(np.maximum(losses @ w_opt - VaR.value, 0)) / (1 - alpha_cvar)) * 100
                sharpe = (port_ret - benchmark_return_mean * 100) / port_vol if port_vol > 0 else 0

                result = {
                    'Portfolio': "-".join(tickers_ord),
                    'Expected Return (%)': port_ret,
                    'Volatility (%)': port_vol,
                    'CVaR (%)': final_cvar,
                    'Sharpe Ratio': sharpe,
                    'Weights': dict(zip(tickers_ord, w_opt))
                }
                real_portfolios.append(result)

                if fallback_result is None or sharpe > fallback_result['Sharpe Ratio']:
                    fallback_result = result

        except Exception as e:
            warnings.warn(f"⚠️ Optimization failed for combo {combo}: {e}")
            continue

    # === 2. Thêm danh mục mô phỏng cho gradient visualization ===
    simulated_portfolios = []
    np.random.seed(2025)
    for combo in valid_combinations:
        tickers = list(combo)
        mu_dict = adj_returns_combinations[combo]
        cov_df = cov_matrix_dict[combo]

        mu = np.array([mu_dict[t] for t in tickers]) / 100
        cov = cov_df.loc[tickers, tickers].values
        if np.any(np.linalg.eigvalsh(cov) < -1e-6):
            continue

        for _ in range(250):  # Số lượng danh mục mô phỏng
            w_rand = np.random.dirichlet(np.ones(len(tickers)))
            port_ret = np.dot(mu, w_rand) * 100
            port_vol = np.sqrt(w_rand.T @ cov @ w_rand) * 100
            sharpe = (port_ret - benchmark_return_mean * 100) / port_vol if port_vol > 0 else 0

            simulated_portfolios.append({
                'Portfolio': 'Simulated',
                'Expected Return (%)': port_ret,
                'Volatility (%)': port_vol,
                'Sharpe Ratio': sharpe
            })

    # === 3. Combine for visualization ===
    df_real = pd.DataFrame(real_portfolios)
    df_sim = pd.DataFrame(simulated_portfolios)
    df_all = pd.concat([df_real, df_sim], ignore_index=True)

    mu_list = df_all['Expected Return (%)'].tolist()
    sigma_list = df_all['Volatility (%)'].tolist()
    sharpe_list = df_all['Sharpe Ratio'].tolist()

    results_ef = (mu_list, sigma_list, sharpe_list)
    hrp_result_dict = {tuple(row['Portfolio'].split("-")): row for _, row in df_real.iterrows()}

    # Fallback nếu không có portfolio thực nào
    if not hrp_result_dict and fallback_result:
        hrp_result_dict[tuple(fallback_result['Portfolio'].split("-"))] = fallback_result

    return hrp_result_dict, results_ef
