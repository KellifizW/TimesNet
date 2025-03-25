import os
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

    # 檢查 preds 的維度
    if preds.ndim == 1:  # 一維數組 (10,)
        future_prices = preds
    elif preds.ndim == 2:  # 二維數組 (1, pred_len)
        future_prices = preds[0]
    else:  # 三維數組 (1, pred_len, features)
        future_prices = preds[0, :, feature_idx if preds.shape[-1] > feature_idx else 0]

    future_dates_flat = [str(date)[:10] for date in future_dates]

    # 過濾交易日
    filtered_dates, trading_mask = filter_trading_days(future_dates_flat)
    filtered_prices = future_prices[trading_mask]

    return filtered_dates, filtered_prices

def load_loss_data(loss_csv_path):
    """加載損失數據並返回 DataFrame"""
    return pd.read_csv(loss_csv_path)

def load_historical_data(historical_data_path, seq_len=15):
    """加載歷史數據並返回最近 seq_len 天的日期和價格（只包含交易日）"""
    df = pd.read_csv(historical_data_path)
    historical_dates = pd.to_datetime(df["Date"]).values[-seq_len:]
    historical_prices = df["Close"].values[-seq_len:]
    historical_dates_flat = [str(date)[:10] for date in historical_dates]

    # 過濾交易日
    filtered_dates, trading_mask = filter_trading_days(historical_dates_flat)
    filtered_prices = historical_prices[trading_mask]

    return filtered_dates, filtered_prices


def load_and_process_test_predict_data(setting, pred_len, historical_data_path, feature_idx=3):
    test_pred_path = os.path.join("results", f"{setting}_test_preds.npy")
    test_true_path = os.path.join("results", f"{setting}_test_trues.npy")
    pred_path = os.path.join("results", f"{setting}_preds.npy")
    dates_path = os.path.join("results", f"{setting}_dates.npy")

    all_pred = np.load(test_pred_path)  # (samples, pred_len, features) or (samples, pred_len)
    all_true = np.load(test_true_path)  # (samples, pred_len, features) or (samples, pred_len)
    pred = np.load(pred_path)           # (samples, pred_len) or (pred_len,)
    pred_dates = np.load(dates_path, allow_pickle=True).flatten()

    # 處理多變量數據的情況
    if all_pred.ndim == 3:
        all_pred = all_pred[:, :, feature_idx]  # 提取指定特徵（例如收盤價）
    if all_true.ndim == 3:
        all_true = all_true[:, :, feature_idx]
    if pred.ndim == 2:
        pred = pred[0]  # 如果是 (1, pred_len)，取第一個樣本
    elif pred.ndim == 3:
        pred = pred[0, :, feature_idx]

    all_pred = all_pred.flatten()
    all_true = all_true.flatten()

    df = pd.read_csv(historical_data_path)
    test_dates_str = df["Date"].iloc[-len(all_true)//pred_len:].tolist()
    all_dates = []
    for start_date in test_dates_str:
        future_dates = generate_trading_days(start_date, pred_len)
        all_dates.extend(future_dates[:pred_len])  # 限制長度為 pred_len
    all_dates = np.array(all_dates[:len(all_pred)])  # 與 all_pred 對齊
    all_dates = np.concatenate([all_dates, pred_dates])

    if len(all_dates) != len(all_pred) + len(pred):
        raise ValueError(f"Data length mismatch: dates={len(all_dates)}, pred={len(all_pred) + len(pred)}, true={len(all_true)}")

    return all_dates, all_pred, all_true