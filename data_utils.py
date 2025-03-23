import numpy as np
from datetime import datetime, timedelta

def flatten_data(data):
    """將多維數組展平為一維數組"""
    return data.flatten()

def load_forecast_data(preds_path, dates_path, feature_idx=3):
    """加載預測數據並返回展平後的日期和價格"""
    preds = np.load(preds_path)
    dates = np.load(dates_path, allow_pickle=True)
    future_dates = dates[0]  # 預測日期
    future_prices = preds[0, :, feature_idx]  # 預測價格（Close）
    future_dates_flat = [str(date[0])[:10] for date in future_dates]  # 展平並轉換日期格式
    return future_dates_flat, future_prices

def load_loss_data(loss_csv_path):
    """加載損失數據並返回 DataFrame"""
    import pandas as pd
    return pd.read_csv(loss_csv_path)

def load_historical_data(historical_data_path, seq_len=15):
    """加載歷史數據並返回最近 seq_len 天的日期和價格"""
    import pandas as pd
    df = pd.read_csv(historical_data_path)
    historical_dates = pd.to_datetime(df["Date"]).values[-seq_len:]
    historical_prices = df["Close"].values[-seq_len:]
    historical_dates_flat = [str(date)[:10] for date in historical_dates]
    return historical_dates_flat, historical_prices

def load_and_process_test_predict_data(test_preds_path, test_trues_path, test_dates_path, pred_preds_path, pred_dates_path, feature_idx=3):
    """加載並處理測試集和預測數據，返回拼接後的日期、預測值和真實值"""
    # 加載測試集數據
    test_preds = np.load(test_preds_path)  # 形狀: (7, 5, 5)
    test_trues = np.load(test_trues_path)  # 形狀: (7, 5, 5)
    test_dates = np.load(test_dates_path, allow_pickle=True)  # 形狀: (7,)

    # 加載預測數據
    pred_preds = np.load(pred_preds_path)  # 形狀: (1, 5, 5)
    pred_dates = np.load(pred_dates_path, allow_pickle=True)  # 形狀: (1, 5)

    # 提取收盤價
    test_pred_close = test_preds[:, :, feature_idx]  # 形狀: (7, 5)
    test_true_close = test_trues[:, :, feature_idx]  # 形狀: (7, 5)
    pred_close = pred_preds[0, :, feature_idx]       # 形狀: (5,)

    # 生成測試集連續日期
    start_date = datetime.strptime(test_dates[0], "%Y-%m-%d")
    test_days = len(test_dates) + 4  # 7 個樣本 + 5 天預測 - 1 = 11 天
    test_date_list = [start_date + timedelta(days=i) for i in range(test_days)]
    test_date_str_list = [d.strftime("%Y-%m-%d") for d in test_date_list]

    # 展平測試集預測（處理重疊，平均值）
    test_pred_flat = np.zeros(test_days)
    test_true_flat = np.zeros(test_days)
    counts = np.zeros(test_days)

    for i in range(len(test_dates)):
        offset = i
        for j in range(5):
            idx = offset + j
            if idx < test_days:
                test_pred_flat[idx] += test_pred_close[i, j]
                test_true_flat[idx] += test_true_close[i, j]
                counts[idx] += 1

    test_pred_flat = test_pred_flat / np.where(counts == 0, 1, counts)
    test_true_flat = test_true_flat / np.where(counts == 0, 1, counts)

    # 處理預測數據日期
    pred_date_str_list = [str(date[0])[:10] for date in pred_dates[0]]

    # 拼接所有數據
    all_dates = test_date_str_list + pred_date_str_list
    all_pred = np.concatenate([test_pred_flat, pred_close])
    all_true = np.concatenate([test_true_flat, [np.nan] * 5])  # 未來 5 天無真實值

    return all_dates, all_pred, all_true