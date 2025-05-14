import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

def run(
    best_portfolio, mu_p, sigma_p, rf, sigma_c, expected_rc, y_capped, y_opt,
    adj_returns_combinations=None, cov_matrix_dict=None, simulate_for_visual=True
):
    st.markdown("### Efficient Frontier with Optimal Complete Portfolio")

    try:
        best_key = list(best_portfolio.keys())[0]
        tickers = list(best_key)
        result = best_portfolio[best_key]
        weights = np.array([result['Weights'][t] for t in tickers])
    except Exception as e:
        st.error(f"❌ Failed to extract best portfolio: {e}")
        return

    # --- Step 1: Get realistic mu and covariance ---
    if adj_returns_combinations and cov_matrix_dict:
        try:
            mu_dict = adj_returns_combinations.get(best_key)
            cov_df = cov_matrix_dict.get(best_key)
            if mu_dict is None or cov_df is None:
                raise ValueError("Missing mu or covariance data for best portfolio key.")
            mu_realistic = np.array([mu_dict[t] for t in tickers]) / 100
            cov = cov_df.loc[tickers, tickers].values
        except Exception as e:
            st.warning(f"⚠️ Fallback: failed to fetch mu/cov accurately: {e}")
            mu_realistic = np.full(len(tickers), result.get('Expected Return (%)', 0) / 100)
            cov = np.outer(weights, weights) * (result.get('Volatility (%)', 0) / 100) ** 2
    else:
        mu_realistic = np.full(len(tickers), result.get('Expected Return (%)', 0) / 100)
        cov = np.outer(weights, weights) * (result.get('Volatility (%)', 0) / 100) ** 2

    # --- Step 2: Simulate Efficient Frontier Portfolios ---
    if simulate_for_visual and len(tickers) > 0:
        n_simulations = 20000  # tăng số lượng để mượt hơn
        np.random.seed(42)
        weights_sim = np.random.dirichlet(np.ones(len(tickers)), size=n_simulations)
        mu_sim = weights_sim @ mu_realistic
        sigma_sim = np.sqrt(np.einsum('ij,jk,ik->i', weights_sim, cov, weights_sim))
        sharpe_sim = (mu_sim - rf) / sigma_sim

        # Loại bỏ điểm không hợp lệ
        valid_mask = (sigma_sim > 1e-5) & (mu_sim > 1e-5) & np.isfinite(sharpe_sim)
        mu_sim, sigma_sim, sharpe_sim = mu_sim[valid_mask], sigma_sim[valid_mask], sharpe_sim[valid_mask]
    else:
        mu_sim, sigma_sim, sharpe_sim = np.array([]), np.array([]), np.array([])

    # --- Step 3: Plotting ---
    fig, ax = plt.subplots(figsize=(10, 7), dpi=100, facecolor="#121212")

    # Gradient scatter theo Sharpe Ratio
    if len(mu_sim) > 0:
        sc = ax.scatter(
            sigma_sim * 100, mu_sim * 100,
            c=sharpe_sim, cmap='plasma',
            s=12, alpha=0.85, edgecolors='none'
        )
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Sharpe Ratio", fontsize=11, color='white')
        cbar.ax.tick_params(labelsize=9, colors='white')
        plt.setp(cbar.ax.get_yticklabels(), color='white')

    # Các điểm đặc trưng
    ax.scatter(0, rf * 100, c='deepskyblue', s=120, label=f"Risk-Free Rate ({rf * 100:.2f}%)", zorder=5)

    ax.scatter(
        sigma_p * 100, mu_p * 100,
        c='red', marker='*', s=200,
        label=f"Optimal Risky Portfolio ({', '.join(tickers)})", edgecolors='black', linewidth=1.0, zorder=10
    )

    ax.scatter(
        sigma_c * 100, expected_rc * 100,
        c='lime', marker='D', s=170,
        label=f"Optimal Complete Portfolio (y={y_capped:.2f})", edgecolors='black', linewidth=1.0, zorder=9
    )

    # Nếu có danh mục đòn bẩy
    if abs(y_opt - y_capped) > 0.01:
        sigma_uncapped = y_opt * sigma_p
        rc_uncapped = y_opt * mu_p + (1 - y_opt) * rf
        ax.scatter(
            sigma_uncapped * 100, rc_uncapped * 100,
            c='magenta', marker='X', s=150,
            label=f"Leveraged Portfolio (y={y_opt:.2f})", edgecolors='black', linewidth=1.0, zorder=8
        )

    # Capital Allocation Line (CAL)
    slope = (mu_p - rf) / sigma_p if sigma_p > 0 else 0
    max_x = max(sigma_sim.max() * 100 * 1.3 if len(sigma_sim) > 0 else sigma_p * 1.5, sigma_p * 100 * 1.5)
    x_cal = np.linspace(0, max_x, 300)
    y_cal = rf * 100 + slope * x_cal
    ax.plot(x_cal, y_cal, 'r--', linewidth=2, label="Capital Allocation Line (CAL)")

    # Format chart
    ax.set_title("Efficient Frontier with Optimal Complete Portfolio", fontsize=16, color='white')
    ax.set_xlabel("Volatility (%)", fontsize=14, color='white')
    ax.set_ylabel("Expected Return (%)", fontsize=14, color='white')
    ax.tick_params(labelsize=11, colors='white')
    ax.set_facecolor("#121212")
    fig.patch.set_facecolor("#121212")
    ax.grid(False)
    ax.legend(facecolor='#1e1e1e', labelcolor='white', fontsize=11, loc="upper left")

    st.pyplot(fig)
