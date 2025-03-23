import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta


def visualize_forecast(setting=None, feature_idx=3):
    # 根據 setting 構建文件路徑
    prefix = f"{setting}_" if setting else ""
    preds_path = os.path.join("results", f"{prefix}preds.npy")
    trues_path = os.path.join("results", f"{prefix}trues.npy")
    dates_path = os.path.join("results", f"{prefix}dates.npy")

    # 讀取預測結果
    preds = np.load(preds_path)
    trues = np.load(trues_path)
    dates = np.load(dates_path)

    # 讀取原始數據
    df = pd.read_csv("data/raw/aapl_daily.csv")
    historical_dates = df["Date"].values[-30:]  # 最近 30 天
    historical_prices = df["Close"].values[-30:]  # 歷史 Close 價格

    # 最近 30 天的預測數據（從測試集中提取）
    pred_dates = dates[-30:]  # 測試集中的最近 30 天日期
    pred_prices = trues[-30:, :, feature_idx]  # 測試集中的真實值（僅取指定特徵，例如 Close）

    # 未來 5 天的預測日期和價格
    last_date = pd.to_datetime(dates[-1])
    future_dates = [(last_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 6)]
    future_prices = preds[-1, :, feature_idx]  # 最後一個預測值（未來 5 天，僅取指定特徵）

    # 合併預測日期和價格
    all_pred_dates = np.concatenate([pred_dates, future_dates])
    all_pred_prices = np.concatenate([pred_prices.flatten(), future_prices])

    # 繪製股價圖表
    fig = go.Figure()

    # 歷史股價
    fig.add_trace(go.Scatter(
        x=historical_dates,
        y=historical_prices,
        mode="lines",
        name="Historical Price (Last 30 Days)"
    ))

    # 預測股價（最近 30 天 + 未來 5 天）
    fig.add_trace(go.Scatter(
        x=all_pred_dates,
        y=all_pred_prices,
        mode="lines",
        name="Predicted Price (Last 30 Days + Next 5 Days)"
    ))

    fig.update_layout(
        title="AAPL Stock Price Forecast",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark"
    )

    os.makedirs("results", exist_ok=True)
    plot_path = os.path.join("results", f"{prefix}forecast_plot.html")
    fig.write_html(plot_path)
    print(f"Forecast plot saved to {plot_path}")


def visualize_loss(setting=None):
    # 根據 setting 構建文件路徑
    prefix = f"{setting}_" if setting else ""
    loss_csv_path = os.path.join("results", f"{prefix}losses.csv")

    # 讀取損失數據
    df = pd.read_csv(loss_csv_path)

    # 繪製損失曲線
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


if __name__ == "__main__":
    # 提供一個示例 setting（與 run.py 中的一致）
    setting = "long_term_forecast_stock_forecast_TimesNet_sl15_pl5_dm90_df90_el2_dropout0.1_test"
    visualize_forecast(setting=setting, feature_idx=3)  # feature_idx=3 表示 Close
    visualize_loss(setting=setting)