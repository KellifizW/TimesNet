import numpy as np
preds = np.load("results/long_term_forecast_stock_forecast_TimesNet_sl15_pl5_dm90_df90_el2_dropout0.1_test_test_preds.npy")
dates = np.load("results/long_term_forecast_stock_forecast_TimesNet_sl15_pl5_dm90_df90_el2_dropout0.1_test_test_dates.npy", allow_pickle=True)
print(f"Test preds shape: {preds.shape}")
print(f"Test dates: {dates}")
print(f"Test preds (Close price): {preds[:, :, 3]}")  # feature_idx=3 是收盤價