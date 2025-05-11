import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def run(mu, cov, rf, mu_p, sigma_p, sigma_c, expected_rc, y_capped, y_opt):
    st.markdown("### Efficient Frontier with CAL and Optimal Portfolios")

    n_assets = len(mu)
    n_sim = 4000
    np.random.seed(42)

    mu_sim, sigma_sim, sharpe_sim = [], [], []

    for _ in range(n_sim):
        w = np.random.dirichlet(np.ones(n_assets) * 0.3)  # Cấu trúc lệch hơn
        mu_noise = mu + np.random.normal(0, 0.002, size=n_assets)
        cov_noise = cov + np.random.normal(0, 0.0008, size=cov.shape)
        cov_noise = (cov_noise + cov_noise.T) / 2  # Giữ đối xứng
        cov_noise = np.clip(cov_noise, 1e-6, None)  # Không âm

        try:
            r = np.dot(w, mu_noise)
            s = np.sqrt(w @ cov_noise @ w)
            sharpe = (r - rf) / s if s > 0 else 0
            mu_sim.append(r * 100)
            sigma_sim.append(s * 100)
            sharpe_sim.append(sharpe)
        except:
            continue

    mu_sim, sigma_sim, sharpe_sim = np.array(mu_sim), np.array(sigma_sim), np.array(sharpe_sim)

    # === Vẽ biểu đồ ===
    fig, ax = plt.subplots(figsize=(10, 7), facecolor='#121212')
    scatter = ax.scatter(sigma_sim, mu_sim, c=sharpe_sim, cmap='viridis', s=10, alpha=0.8)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Sharpe Ratio", color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.get_yticklabels(), color='white')

    # Risk-Free Point
    ax.scatter(0, rf * 100, c='blue', s=100, marker='o', label=f"Risk-Free Rate ({rf*100:.2f}%)")

    # Optimal Risky Portfolio
    ax.scatter(sigma_p * 100, mu_p * 100, c='red', marker='*', s=220, label="Optimal Risky Portfolio")

    # CAL Line
    slope = (mu_p - rf) / sigma_p
    x_cal = np.linspace(0, sigma_sim.max() * 1.1, 100)
    y_cal = rf + slope * (x_cal / 100)
    ax.plot(x_cal, y_cal * 100, 'r--', linewidth=2, label="Capital Allocation Line (CAL)")

    # Complete Portfolio
    ax.scatter(sigma_c * 100, expected_rc * 100, c='lime', marker='D', s=160,
               label=f"Complete Portfolio (y = {y_capped:.2f})")

    # Leveraged if applicable
    if abs(y_opt - y_capped) > 1e-3:
        sigma_lev = y_opt * sigma_p
        mu_lev = y_opt * mu_p + (1 - y_opt) * rf
        ax.scatter(sigma_lev * 100, mu_lev * 100, c='magenta', marker='X', s=150,
                   label=f"Leveraged Portfolio (y = {y_opt:.2f})")

    # Style
    ax.set_facecolor('#121212')
    fig.patch.set_facecolor('#121212')
    ax.set_title("Efficient Frontier with CAL and Optimal Portfolios", color='white', fontsize=14)
    ax.set_xlabel("Volatility (%)", color='white')
    ax.set_ylabel("Expected Return (%)", color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1e1e1e', labelcolor='white', fontsize=9, loc='upper left')
    ax.grid(False)

    st.pyplot(fig)
