import numpy as np
import os
import plotly.graph_objects as go
from data_utils import (
    load_forecast_data,
    load_loss_data,
    load_historical_data,
    load_and_process_test_predict_data
)

def visualize_forecast(setting=None, feature_idx=3):
    prefix = f"{setting}_" if setting else ""
    preds_path = os.path.join("results", f"{prefix}preds.npy")
    dates_path = os.path.join("results", f"{prefix}dates.npy")

    if not os.path.exists(preds_path) or not os.path.exists(dates_path):
        print(f"Error: Prediction files ({preds_path} or {dates_path}) not found.")
        return

    future_dates, future_prices = load_forecast_data(preds_path, dates_path, feature_idx)

    # 打印預測結果
    print(f"\nFuture {len(future_dates)} Days Stock Price Forecast (Close Price):")
    for date, price in zip(future_dates, future_prices):
        print(f"{date}: ${price:.2f}")

    # 保存預測結果
    os.makedirs("results", exist_ok=True)
    forecast_path = os.path.join("results", f"{prefix}future_forecast.txt")
    with open(forecast_path, "w") as f:
        f.write(f"Future {len(future_dates)} Days Stock Price Forecast (Close Price):\n")
        for date, price in zip(future_dates, future_prices):
            f.write(f"{date}: ${price:.2f}\n")
    print(f"Forecast saved to {forecast_path}")

def visualize_loss(setting=None):
    prefix = f"{setting}_" if setting else ""
    loss_csv_path = os.path.join("results", f"{prefix}losses.csv")

    if not os.path.exists(loss_csv_path):
        print(f"Error: Loss file {loss_csv_path} not found.")
        return

    df = load_loss_data(loss_csv_path)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Epoch"], y=df["Train_Loss"], mode="lines", name="Train Loss"))
    fig.add_trace(go.Scatter(x=df["Epoch"], y=df["Valid_Loss"], mode="lines", name="Valid Loss"))

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

    if not os.path.exists(preds_path) or not os.path.exists(dates_path):
        print(f"Error: Prediction files ({preds_path} or {dates_path}) not found.")
        return

    future_dates, future_prices = load_forecast_data(preds_path, dates_path, feature_idx)

    # 加載歷史數據
    if historical_data_path is None:
        historical_data_path = "data/raw/aapl_daily.csv"
    print(f"Loading historical data from {historical_data_path}")
    if not os.path.exists(historical_data_path):
        print(f"Error: Historical data file {historical_data_path} not found. Skipping comparison plot.")
        return

    historical_dates, historical_prices = load_historical_data(historical_data_path)

    # 繪製對比圖
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical_dates, y=historical_prices, mode="lines", name="Historical Close Price", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=future_dates, y=future_prices, mode="lines", name="Forecasted Close Price", line=dict(color="red", dash="dash")))

    fig.update_layout(
        title="Historical vs Forecasted Stock Prices",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark"
    )

    os.makedirs("results", exist_ok=True)
    comparison_plot_path = os.path.join("results", f"{prefix}comparison_plot.html")
    fig.write_html(comparison_plot_path)
    print(f"Comparison plot saved to {comparison_plot_path}")

def visualize_test_and_predict(setting=None, feature_idx=3):
    prefix = f"{setting}_"
    test_preds_path = os.path.join("results", f"{prefix}test_preds.npy")
    test_trues_path = os.path.join("results", f"{prefix}test_trues.npy")
    test_dates_path = os.path.join("results", f"{prefix}test_dates.npy")
    pred_preds_path = os.path.join("results", f"{prefix}preds.npy")
    pred_dates_path = os.path.join("results", f"{prefix}dates.npy")

    if not all(os.path.exists(p) for p in [test_preds_path, test_trues_path, test_dates_path, pred_preds_path, pred_dates_path]):
        print(f"Error: Required files missing for {setting}.")
        return

    all_dates, all_pred, all_true = load_and_process_test_predict_data(
        test_preds_path, test_trues_path, test_dates_path, pred_preds_path, pred_dates_path, feature_idx
    )

    # 繪製圖表
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=all_dates, y=all_true, mode='lines+markers', name='True Close Price', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=all_dates, y=all_pred, mode='lines+markers', name='Predicted Close Price', line=dict(color='red', dash='dash')))

    fig.update_layout(
        title="Test Set and Future Prediction: Close Price",
        xaxis_title="Date",
        yaxis_title="Close Price (USD)",
        legend_title="Legend",
        template="plotly_dark"
    )

    os.makedirs("results", exist_ok=True)
    fig.write_html(f"results/{prefix}test_and_predict_comparison.html")
    print(f"Test and predict comparison plot saved to results/{prefix}test_and_predict_comparison.html")

    # 打印每日股價
    print("\nTest Set and Future Daily Close Price Forecast:")
    for date, pred, true in zip(all_dates, all_pred, all_true):
        true_str = f"${true:.2f}" if not np.isnan(true) else "N/A"
        print(f"{date}: Predicted = ${pred:.2f}, True = {true_str}")

if __name__ == "__main__":
    setting = "long_term_forecast_stock_forecast_TimesNet_sl15_pl5_dm90_df90_el2_dropout0.1_test"
    visualize_forecast(setting=setting, feature_idx=3)
    visualize_loss(setting=setting)
    visualize_comparison(setting=setting, feature_idx=3, historical_data_path="data/raw/aapl_daily.csv")
    visualize_test_and_predict(setting=setting, feature_idx=3)