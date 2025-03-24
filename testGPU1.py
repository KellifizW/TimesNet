import numpy as np
test_trues = np.load("results/long_term_forecast_stock_forecast_TimesNet_sl15_pl5_dm90_df90_el2_dropout0.1_test_test_trues.npy")
print(test_trues[:, 3])  # feature_idx=3