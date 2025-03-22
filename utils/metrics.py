import numpy as np

def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def directional_accuracy(pred, true):
    # 計算方向準確率
    direction_true = (true[1:] - true[:-1]) > 0
    direction_pred = (pred[1:] - pred[:-1]) > 0
    return np.mean(direction_true == direction_pred)