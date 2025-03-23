import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta


def visualize_forecast():
    # 讀取預測結果
    preds = np.load("results/preds.npy")
    trues = np.load("results/trues.npy")
    dates = np.load("results/dates.npy")

    # 讀取原始數據
    df = pd.read_csv("data/raw/aapl_daily.csv")
    historical_dates = df["Date"].values[-30:]  # 最近 30 天
    historical_prices = df["Close"].values[-30:]

    # 最近 30 天的預測數據（從測試集中提取）
    pred_dates = dates[-30:]  # 測試集中的最近 30 天日期
    pred_prices = trues[-30:]  # 測試集中的真實值（作為預測的起點）

    # 未來 5 天的預測日期和價格
    last_date = pd.to_datetime(dates[-1])
    future_dates = [(last_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 6)]
    future_prices = preds[-1]  # 最後一個預測值（未來 5 天）

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
    fig.write_html("results/forecast_plot.html")
    print("Forecast plot saved to results/forecast_plot.html")


def visualize_loss():
    # 讀取損失數據
    df = pd.read_csv("results/losses.csv")

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

    fig.write_html("results/loss_plot.html")
    print("Loss plot saved to results/loss_plot.html")


if __name__ == "__main__":
    visualize_forecast()
    visualize_loss()