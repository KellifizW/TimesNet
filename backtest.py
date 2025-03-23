import pandas as pd
import numpy as np
import torch
import os
from datetime import datetime, timedelta

def backtest(setting, model, data_loader, device, seq_len, pred_len, feature_idx=3):
    try:
        model.eval()
        preds, trues, dates = [], [], []

        # 滾動預測
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                mean = batch["mean"].to(device)
                std = batch["std"].to(device)

                outputs = model(x, None, None, None)
                outputs = outputs * std + mean
                y = y * std + mean

                outputs = outputs.cpu().numpy()
                y = y.cpu().numpy()

                preds.append(outputs)
                trues.append(y)

                # 處理 date 欄位（單個字符串）
                start_date = pd.to_datetime(batch["date"][0])  # batch["date"] 是一個列表，例如 ["2025-03-07", ...]
                batch_dates = [(start_date + timedelta(days=j)).strftime("%Y-%m-%d") for j in range(pred_len)]
                dates.append(batch_dates)

        preds = np.concatenate(preds, axis=0)  # 形狀 (num_samples, pred_len, num_features)
        trues = np.concatenate(trues, axis=0)  # 形狀 (num_samples, pred_len, num_features)
        dates = np.array(dates)  # 形狀 (num_samples, pred_len)

        # 展平數據以計算誤差（僅針對指定特徵，例如 Close）
        preds_flat = preds[:, :, feature_idx].flatten()
        trues_flat = trues[:, :, feature_idx].flatten()
        dates_flat = dates.flatten()

        # 計算誤差指標
        mse = np.mean((preds_flat - trues_flat) ** 2)
        mae = np.mean(np.abs(preds_flat - trues_flat))
        mape = np.mean(np.abs((preds_flat - trues_flat) / trues_flat)) * 100
        print(f"Backtest MSE: {mse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")

        # 計算方向準確率
        direction_correct = 0
        for i in range(1, len(preds_flat)):
            pred_change = preds_flat[i] - trues_flat[i-1]
            true_change = trues_flat[i] - trues_flat[i-1]
            if (pred_change > 0 and true_change > 0) or (pred_change < 0 and true_change < 0):
                direction_correct += 1
        direction_accuracy = direction_correct / (len(preds_flat) - 1) * 100
        print(f"Direction Accuracy: {direction_accuracy:.2f}%")

        # 模擬交易策略（簡單策略：預測上漲則買入，預測下跌則賣出，添加止損）
        capital = 10000
        position = 0
        stop_loss = 0.05
        trades = []
        for i in range(1, len(preds_flat)):
            pred_price = preds_flat[i]
            true_price = trues_flat[i]
            prev_price = trues_flat[i-1]

            if position > 0 and (prev_price - true_price) / prev_price > stop_loss:
                capital = position * true_price
                position = 0
                trades.append(f"{dates_flat[i]}: Stop Loss at {true_price:.2f}, Capital: {capital:.2f}")
            elif pred_price > prev_price and position == 0:
                position = capital / true_price
                capital = 0
                trades.append(f"{dates_flat[i]}: Buy at {true_price:.2f}, Position: {position:.2f}")
            elif pred_price < prev_price and position > 0:
                capital = position * true_price
                position = 0
                trades.append(f"{dates_flat[i]}: Sell at {true_price:.2f}, Capital: {capital:.2f}")

        final_value = capital + position * trues_flat[-1]
        roi = (final_value - 10000) / 10000 * 100
        print(f"Final Value: ${final_value:.2f}, ROI: {roi:.2f}%")

        os.makedirs("results", exist_ok=True)
        prefix = f"{setting}_"
        backtest_path = os.path.join("results", f"{prefix}backtest_result.txt")
        with open(backtest_path, "w") as f:
            f.write(f"Backtest MSE: {mse:.4f}\n")
            f.write(f"Backtest MAE: {mae:.4f}\n")
            f.write(f"Backtest MAPE: {mape:.2f}%\n")
            f.write(f"Direction Accuracy: {direction_accuracy:.2f}%\n")
            f.write(f"Final Value: ${final_value:.2f}\n")
            f.write(f"ROI: {roi:.2f}%\n")
            f.write("\nTrades:\n")
            for trade in trades:
                f.write(f"{trade}\n")
        print(f"Backtest results saved to {backtest_path}")

    except Exception as e:
        print(f"Error in backtest: {str(e)}")
        raise