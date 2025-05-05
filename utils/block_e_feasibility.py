# utils/block_e_feasibility.py

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
import warnings
import streamlit as st

def run(adj_returns_combinations, cov_matrix_dict):
    valid_combinations = []
    invalid_log = []

    target_combinations = list(adj_returns_combinations.keys())

    for combo in target_combinations:
        tickers = combo.split('-')

        # --- Kiểm tra vector lợi suất kỳ vọng ---
        try:
            mu = np.array([adj_returns_combinations[combo][t] for t in tickers]) / 100
        except KeyError:
            invalid_log.append((combo, "❌ Missing expected return"))
            continue

        # --- Kiểm tra ma trận hiệp phương sai ---
        try:
            cov_df = cov_matrix_dict[combo].copy()
            cov = cov_df.loc[tickers, tickers].values
        except Exception as e:
            invalid_log.append((combo, f"❌ Covariance matrix error: {e}"))
            continue

        # --- Các kiểm tra logic ---
        if np.any(np.isnan(mu)) or np.any(np.isinf(mu)):
            invalid_log.append((combo, "❌ Return vector contains NaN or Inf"))
            continue

        if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
            invalid_log.append((combo, "❌ Covariance matrix contains NaN or Inf"))
            continue

        if np.all(mu <= 0):
            invalid_log.append((combo, "❌ All expected returns <= 0"))
            continue

        try:
            eigvals = np.linalg.eigvalsh(cov)
            if np.any(eigvals < -1e-6):
                invalid_log.append((combo, "❌ Covariance matrix not PSD"))
                continue
        except LinAlgError as e:
            invalid_log.append((combo, f"❌ Eigendecomposition failed: {e}"))
            continue

        # --- Nếu vượt qua mọi kiểm tra ---
        valid_combinations.append(combo)

    # --- Summary ---
    st.markdown("### 📊 Portfolio Feasibility Check Summary")
    st.write(f"✅ **Valid combinations**: {len(valid_combinations)}")
    st.write(f"❌ **Invalid combinations**: {len(invalid_log)}")

    if invalid_log:
        with st.expander("🔍 Details of Invalid Portfolios"):
            for combo, reason in invalid_log:
                st.write(f"• `{combo}` → {reason}")

    return valid_combinations
