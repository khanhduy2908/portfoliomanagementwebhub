def run_stress_test_visualizations(stress_results_macro, stress_results_industry, stress_results_idio):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # --- 1. Macro Shock Visualization ---
    df_macro = pd.DataFrame(stress_results_macro)
    df_macro.set_index('Scenario', inplace=True)
    df_macro.T.plot(kind='bar', figsize=(12, 6), colormap='RdBu', edgecolor='black')
    plt.title("Portfolio Performance Under Macro Stress Scenarios")
    plt.ylabel("Portfolio Return (%)")
    plt.xlabel("Assets")
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    # --- 2. Industry Shock Visualization ---
    df_industry = pd.DataFrame(stress_results_industry)
    df_industry.set_index('Scenario', inplace=True)
    df_industry.T.plot(kind='bar', figsize=(12, 6), colormap='coolwarm', edgecolor='black')
    plt.title("Portfolio Performance Under Industry Stress Scenarios")
    plt.ylabel("Portfolio Return (%)")
    plt.xlabel("Assets")
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    # --- 3. Idiosyncratic Shock Visualization ---
    df_idio = pd.DataFrame(stress_results_idio)
    df_idio.set_index('Ticker', inplace=True)
    df_idio.plot(kind='bar', figsize=(12, 6), color='gray', edgecolor='black')
    plt.title("Impact of Idiosyncratic Shocks on Individual Assets")
    plt.ylabel("Simulated Loss (%)")
    plt.xlabel("Ticker")
    plt.grid(False)
    plt.tight_layout()
    plt.show()
