# Institutional Portfolio Optimization Platform

A professional-grade web application for portfolio optimization using modern techniques inspired by investment banks and institutional asset managers.

## 🧩 System Overview

This app runs a full pipeline from data ingestion to portfolio construction and evaluation, structured into modular blocks:

| Block | Name                            | Description                                                                |
| ----- | ------------------------------- | -------------------------------------------------------------------------- |
| A     | Data Loading & Preprocessing    | Load historical stock data, compute monthly returns                        |
| B     | Factor Ranking & Filtering      | Score and rank stocks based on multi-factor models                         |
| C     | Covariance Estimation           | Estimate robust covariance matrix using GARCH + Ledoit-Wolf                |
| D     | Return Forecasting              | Predict future returns using ensemble stacking (LightGBM, XGBoost, TabNet) |
| E     | Portfolio Feasibility Check     | Validate portfolio candidates for modeling integrity                       |
| F     | Walkforward Backtest            | Evaluate model generalization across time windows                          |
| G     | Optimization (HRP + Soft CVaR)  | Optimize portfolio using hierarchical clustering + CVaR constraint         |
| H     | Complete Portfolio Construction | Allocate capital based on utility theory and CAL line                      |
| I     | Performance Evaluation          | Measure Sharpe, Sortino, Drawdown, Alpha/Beta, CAGR                        |
| E1/E2 | Risk-Return Visualizations      | Visualize asset risk vs return and benchmark comparisons                   |
| J     | Multi-Layer Stress Testing      | Test resilience under macro, sector, and asset-level shocks                |

---

## 🚀 How to Run

### 1. Clone Repository

```bash
git clone https://github.com/your-org/portfolio-optimization-app.git
cd portfolio-optimization-app
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch App

```bash
streamlit run app.py
```

---

## 📁 Folder Structure

```
├── app.py
├── config.py
├── requirements.txt
├── utils/
│   ├── valid_tickers.txt
│   ├── block_a_data.py
│   ├── block_b_factor.py
│   ├── block_c_covariance.py
│   ├── block_d_forecast.py
│   ├── block_e_feasibility.py
│   ├── block_f_backtest.py
│   ├── block_g_optimization.py
│   ├── block_h_complete_portfolio.py
│   ├── block_i_performance_analysis.py
│   ├── block_e1_visualization.py
│   ├── block_e2_visualization.py
│   └── block_j_stress_testing.py
```

---

## 📌 Configuration

All shared configuration (tickers, dates, risk-free rate, capital, risk aversion, etc.) are managed in `config.py` and dynamically set by the UI.

---

## 📬 Contact

For questions or contributions, reach out to the author or create an issue in the repository.
