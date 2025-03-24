import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

def flatten_data(data):
    """將多維數組展平為一維數組"""
    return data.flatten()

def filter_trading_days(dates, start_date=None, end_date=None):
    """
    過濾掉非交易日（週末和美國聯邦節假日），只保留交易日。
    參數：
        dates: 日期列表（可以是字符串或 datetime 格式）
        start_date: 可選，指定日期範圍的開始日期
        end_date: 可選，指定日期範圍的結束日期
    返回：
        filtered_dates: 過濾後的交易日日期（numpy 陣列）
        mask: 布林遮罩，用於過濾其他對應的數據
    """
    dates = pd.to_datetime(dates)

    # 檢查是否為工作日（星期一到星期五）
    is_weekday = dates.weekday < 5

    # 檢查是否為美國聯邦節假日
    if start_date is None:
        start_date = dates.min()
    if end_date is None:
        end_date = dates.max()
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=start_date, end=end_date)
    is_not_holiday = ~dates.isin(holidays)

    # 結合兩個條件：必須是工作日且不是節假日
    trading_day_mask = is_weekday & is_not_holiday

    # 過濾日期
    filtered_dates = dates[trading_day_mask].values

    return filtered_dates, trading_day_mask

def generate_trading_days(start_date, num_days):
    """
    生成指定數量的連續交易日。
    參數：
        start_date: 開始日期（字符串或 datetime 格式）
        num_days: 需要生成的交易日數量
    返回：
        trading_days: 交易日列表（字符串格式）
    """
    # 確保 start_date 是單一的 pandas.Timestamp 對象
    if isinstance(start_date, (np.ndarray, pd.Index)):
        start_date = start_date[0]  # 取第一個元素
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    elif not isinstance(start_date, pd.Timestamp):
        start_date = pd.to_datetime(start_date)

    trading_days = []
    current_date = start_date
    while len(trading_days) < num_days:
        # 檢查是否為交易日
        is_weekday = current_date.weekday() < 5
        is_not_holiday = True
        if is_weekday:
            cal = USFederalHolidayCalendar()
            holidays = cal.holidays(start=current_date, end=current_date)
            is_not_holiday = current_date not in holidays
        if is_weekday and is_not_holiday:
            trading_days.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)
    return trading_days

def load_forecast_data(preds_path, dates_path, feature_idx=3):
    """加載預測數據並返回展平後的日期和價格（只包含交易日）"""
    preds = np.load(preds_path)
    dates = np.load(dates_path, allow_pickle=True)
    future_dates = dates[0]  # 預測日期
    future_prices = preds[0, :, feature_idx]  # 預測價格（Close）
    future_dates_flat = [str(date)[:10] for date in future_dates]  # 展平並轉換日期格式

    # 過濾交易日
    filtered_dates, trading_mask = filter_trading_days(future_dates_flat)
    filtered_prices = future_prices[trading_mask]

    return filtered_dates, filtered_prices

def load_loss_data(loss_csv_path):
    """加載損失數據並返回 DataFrame"""
    import pandas as pd
    return pd.read_csv(loss_csv_path)

def load_historical_data(historical_data_path, seq_len=15):
    """加載歷史數據並返回最近 seq_len 天的日期和價格（只包含交易日）"""
    import pandas as pd
    df = pd.read_csv(historical_data_path)
    historical_dates = pd.to_datetime(df["Date"]).values[-seq_len:]
    historical_prices = df["Close"].values[-seq_len:]
    historical_dates_flat = [str(date)[:10] for date in historical_dates]

    # 過濾交易日
    filtered_dates, trading_mask = filter_trading_days(historical_dates_flat)
    filtered_prices = historical_prices[trading_mask]

    return filtered_dates, filtered_prices

def load_and_process_test_predict_data(test_preds_path, test_trues_path, test_dates_path, pred_preds_path,
                                       pred_dates_path, feature_idx=3):
    """加載並處理測試集和預測數據，返回拼接後的日期、預測值和真實值（只包含交易日）"""
    # 加載測試集數據
    test_preds = np.load(test_preds_path)  # 形狀: (num_samples, pred_len, features)
    test_trues = np.load(test_trues_path)  # 形狀: (num_samples, pred_len, features)
    test_dates = np.load(test_dates_path, allow_pickle=True)  # 形狀: (num_samples,)

    # 加載預測數據
    pred_preds = np.load(pred_preds_path)  # 形狀: (1, pred_len, features)
    pred_dates = np.load(pred_dates_path, allow_pickle=True)  # 形狀: (1, pred_len)

    # 動態獲取 pred_len
    pred_len = test_preds.shape[1]  # 從 test_preds 的形狀中獲取 pred_len

    # 提取收盤價
    test_pred_close = test_preds[:, :, feature_idx]  # 形狀: (num_samples, pred_len)
    test_true_close = test_trues[:, :, feature_idx]  # 形狀: (num_samples, pred_len)
    pred_close = pred_preds[0, :, feature_idx]  # 形狀: (pred_len,)

    # 將 test_dates 轉換為字符串格式
    test_dates_str = [str(date) for date in test_dates]

    # 過濾 test_dates 中的交易日
    filtered_test_dates, trading_mask = filter_trading_days(test_dates_str)
    filtered_test_dates = [str(date)[:10] for date in filtered_test_dates]

    # 確保 test_dates 只包含交易日
    filtered_test_dates_set = set(filtered_test_dates)
    valid_indices = [i for i, date in enumerate(test_dates_str) if date in filtered_test_dates_set]
    test_dates = test_dates[valid_indices]
    test_pred_close = test_pred_close[valid_indices]
    test_true_close = test_true_close[valid_indices]

    if len(test_dates) == 0:
        raise ValueError("No valid trading days found in test_dates after filtering.")

    # 修正 test_days 的計算：只需要涵蓋從第一個日期到最後一個日期 + pred_len 的範圍
    start_date = test_dates[0]
    end_date = test_dates[-1]
    # 計算日期範圍（從 start_date 到 end_date 的交易日數）
    all_dates_in_range = generate_trading_days(start_date, 1000)  # 生成一個較長的日期列表以涵蓋範圍
    start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
    test_date_range = [d for d in all_dates_in_range if start_date_dt <= datetime.strptime(d, "%Y-%m-%d") <= end_date_dt]
    test_days = len(test_date_range) + pred_len  # 加上預測長度
    test_date_str_list = generate_trading_days(start_date, test_days)

    # 展平測試集預測（處理重疊，平均值）
    test_pred_flat = np.zeros(test_days)
    test_true_flat = np.zeros(test_days)
    counts = np.zeros(test_days)

    # 將 test_dates 轉換為 datetime 格式，以便比較
    test_dates_dt = [datetime.strptime(date, "%Y-%m-%d") for date in test_dates]
    test_date_str_list_dt = [datetime.strptime(date, "%Y-%m-%d") for date in test_date_str_list]

    for i in range(len(test_dates)):
        try:
            start_idx = test_date_str_list_dt.index(test_dates_dt[i])
            for j in range(pred_len):  # 使用動態的 pred_len
                idx = start_idx + j
                if idx < test_days:
                    test_pred_flat[idx] += test_pred_close[i, j]
                    test_true_flat[idx] += test_true_close[i, j]
                    counts[idx] += 1
        except ValueError as e:
            print(f"警告：日期 {test_dates_dt[i]} 不在生成的交易日列表中，跳過此數據點。")
            continue

    test_pred_flat = test_pred_flat / np.where(counts == 0, 1, counts)
    test_true_flat = test_true_flat / np.where(counts == 0, 1, counts)

    # 修剪數據：只保留有數據的部分
    last_non_zero_idx = np.max(np.where(counts > 0)[0]) if np.any(counts > 0) else 0
    test_date_str_list = test_date_str_list[:last_non_zero_idx + 1]
    test_pred_flat = test_pred_flat[:last_non_zero_idx + 1]
    test_true_flat = test_true_flat[:last_non_zero_idx + 1]

    # 處理預測數據日期
    pred_date_str_list = [str(date)[:10] for date in pred_dates[0]]
    filtered_pred_dates, pred_trading_mask = filter_trading_days(pred_date_str_list)
    filtered_pred_close = pred_close[pred_trading_mask]

    # 拼接所有數據
    all_dates = test_date_str_list + list(filtered_pred_dates)
    all_pred = np.concatenate([test_pred_flat, filtered_pred_close])
    all_true = np.concatenate([test_true_flat, [np.nan] * len(filtered_pred_dates)])  # 未來日期無真實值

    return all_dates, all_pred, all_true