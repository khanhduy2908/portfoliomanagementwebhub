tickers = []
benchmark_symbol = "VNINDEX"
start_date = None
end_date = None
rf_annual = 0.09
rf = 0.0075
total_capital = 750_000_000
A = 15

CVaR_ALPHA = 0.95
CVaR_SOFT_LIMIT = 6.5
Y_MIN = 0.6
Y_MAX = 0.9

TABNET_PARAMS = {
    'seed': 42,
    'max_epochs': 100,
    'patience': 10,
    'batch_size': 256,
    'virtual_batch_size': 128,
    'eval_metric': ['mae'],
}

STACKING_FOLDS = 5
LOOKBACK_WINDOW = 12
MIN_SAMPLE_SIZE = 100

L2_PENALTY = 0.01
LAMBDA_CVAR = 5

MC_SIMULATIONS = 10000
CONFIDENCE_LEVEL = 0.95
T_DIST_DF = 4
INTEREST_RATE_SHOCK = -0.15
INFLATION_SHOCK = -0.10
HISTORICAL_SHOCK = -0.25

CVaR_ALPHA = 0.95
LAMBDA_CVaR = 5
BETA_L2 = 0.01
CVaR_SOFT_LIMIT = 6.5
N_SIMULATIONS = 20000
SEED = 42
SOLVERS = ['SCS', 'ECOS']
