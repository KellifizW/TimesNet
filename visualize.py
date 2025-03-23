import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta


def visualize_forecast(setting=None, feature_idx=3):
    prefix = f"{setting}_" if setting else ""
    preds_path = os.path.join("results", f"{prefix}preds.npy")
    dates_path = os.path.join("results", f"{prefix}dates.npy")

    # 檢查文件是否存在
    if not os.path.exists(preds_path) or not os.path.exists(dates_path):
        print(f"Error: Prediction files ({preds_path} or {dates_path}) not found.")
        return

    preds = np.load(preds_path)
    dates = np.load(dates_path)

    # 確保數據格式正確，扁平化二維日期數組
    future_dates = dates[0]  # 預測日期
    future_prices = preds[0, :, feature_idx]  # 預測價格（Close）

    # 將二維數組扁平化並轉換為字符串格式 (YYYY-MM-DD)
    future_dates = [str(date[0])[:10] for date in future_dates]  # 提取嵌套列表中的第一個元素

    # 打印預測結果
    print("\nFuture 5 Days Stock Price Forecast (Close Price):")
    for date, price in zip(future_dates, future_prices):
        print(f"{date}: ${price:.2f}")

    # 保存預測結果
    os.makedirs("results", exist_ok=True)
    forecast_path = os.path.join("results", f"{prefix}future_forecast.txt")
    with open(forecast_path, "w") as f:
        f.write("Future 5 Days Stock Price Forecast (Close Price):\n")
        for date, price in zip(future_dates, future_prices):
            f.write(f"{date}: ${price:.2f}\n")
    print(f"Forecast saved to {forecast_path}")


def visualize_loss(setting=None):
    prefix = f"{setting}_" if setting else ""
    loss_csv_path = os.path.join("results", f"{prefix}losses.csv")

    # 檢查文件是否存在
    if not os.path.exists(loss_csv_path):
        print(f"Error: Loss file {loss_csv_path} not found.")
        return

    df = pd.read_csv(loss_csv_path)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Epoch"],
        y=df["Train_Loss"],
        mode="lines",
        name="Train Loss"
    ))

    fig.add_trace(go.Scatter(
        x=df["Epoch"],
        y=df["Valid_Loss"],
        mode="lines",
        name="Valid Loss"
    ))

    fig.update_layout(
        title="Training and Validation Loss",
        xaxis_title="Epoch",
        yaxis_title="Loss (MSE)",
        template="plotly_dark"
    )

    plot_path = os.path.join("results", f"{prefix}loss_plot.html")
    fig.write_html(plot_path)
    print(f"Loss plot saved to {plot_path}")


def visualize_comparison(setting=None, feature_idx=3, historical_data_path=None):
    prefix = f"{setting}_" if setting else ""
    preds_path = os.path.join("results", f"{prefix}preds.npy")
    dates_path = os.path.join("results", f"{prefix}dates.npy")

    print(f"Starting visualize_comparison for {setting}...")

    # 加載預測數據
    if not os.path.exists(preds_path) or not os.path.exists(dates_path):
        print(f"Error: Prediction files ({preds_path} or {dates_path}) not found.")
        return

    preds = np.load(preds_path)
    dates = np.load(dates_path)
    future_dates = dates[0]  # 預測日期
    future_prices = preds[0, :, feature_idx]  # 預測價格（Close）

    # 調試：檢查原始日期數據
    print(f"Raw future_dates: {future_dates}")

    # 將二維數組扁平化並轉換為字符串格式 (YYYY-MM-DD)
    future_dates = [str(date[0])[:10] for date in future_dates]  # 提取嵌套列表中的第一個元素
    print(f"Processed future_dates: {future_dates}")  # 調試信息

    # 加載歷史數據
    if historical_data_path is None:
        historical_data_path = "data/raw/aapl_daily.csv"  # 默認使用 aapl_daily.csv
    print(f"Loading historical data from {historical_data_path}")
    if not os.path.exists(historical_data_path):
        print(f"Error: Historical data file {historical_data_path} not found. Skipping comparison plot.")
        return

    historical_df = pd.read_csv(historical_data_path)
    historical_dates = pd.to_datetime(historical_df["Date"]).values
    historical_prices = historical_df["Close"].values

    # 確保歷史數據足夠長，並選取最後 15 天 (seq_len=15)
    sl = 15  # 固定序列長度
    if len(historical_prices) < sl:
        print(f"Error: Historical data length ({len(historical_prices)}) is less than sequence length ({sl}).")
        return
    historical_dates_subset = historical_dates[-sl:]
    historical_prices_subset = historical_prices[-sl:]

    # 將歷史日期轉換為字符串格式 (YYYY-MM-DD)
    historical_dates_subset = [str(date)[:10] for date in historical_dates_subset]
    print(f"Historical dates: {historical_dates_subset}")  # 調試信息

    # 繪製對比圖
    fig = go.Figure()

    # 添加歷史股價線
    fig.add_trace(go.Scatter(
        x=historical_dates_subset,
        y=historical_prices_subset,
        mode="lines",
        name="Historical Close Price",
        line=dict(color="blue")
    ))

    # 添加預測股價線
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_prices,
        mode="lines",
        name="Forecasted Close Price",
        line=dict(color="red", dash="dash")
    ))

    fig.update_layout(
        title="Historical vs Forecasted Stock Prices",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark"
    )

    # 保存圖表
    os.makedirs("results", exist_ok=True)
    comparison_plot_path = os.path.join("results", f"{prefix}comparison_plot.html")
    fig.write_html(comparison_plot_path)
    print(f"Comparison plot saved to {comparison_plot_path}")


if __name__ == "__main__":
    setting = "long_term_forecast_stock_forecast_TimesNet_sl15_pl5_dm90_df90_el2_dropout0.1_test"
    visualize_forecast(setting=setting, feature_idx=3)
    visualize_loss(setting=setting)
    visualize_comparison(setting=setting, feature_idx=3, historical_data_path="data/raw/aapl_daily.csv")