import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from datetime import datetime, timedelta


class StockDataset(Dataset):
    def __init__(self, data_path, seq_len=15, pred_len=5, split="train", train_ratio=0.7, valid_ratio=0.2):
        self.split = split
        df = pd.read_csv(data_path)
        self.features = ['Open', 'High', 'Low', 'Close', 'Volume']
        self.data = df[self.features].values.astype(float)
        self.dates = df["Date"].values

        self.enc_in = self.data.shape[1]
        self.seq_len = seq_len
        self.pred_len = pred_len

        total_len = len(self.data)
        train_end = int(total_len * train_ratio)
        valid_end = int(total_len * (train_ratio + valid_ratio))

        if split == "train":
            self.start_idx = 0
            self.end_idx = train_end
        elif split == "val":
            self.start_idx = train_end
            self.end_idx = valid_end
        elif split == "test":
            self.start_idx = valid_end
            self.end_idx = total_len
        elif split == "predict":
            self.start_idx = total_len - seq_len
            self.end_idx = total_len
        else:
            raise ValueError(f"Unknown split: {split}")

        self.data = self.data[self.start_idx:self.end_idx]
        self.dates = self.dates[self.start_idx:self.end_idx]

    def __len__(self):
        if self.split == "predict":
            return 1
        return self.end_idx - self.start_idx - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        if self.split == "predict":
            idx = 0  # predict 模式下只有一個樣本
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len] if self.split != "predict" else np.zeros(
            (self.pred_len, self.data.shape[1]))

        # 在非 predict 模式下，date 只需要一個時間點（序列最後一天）
        if self.split != "predict":
            date = self.dates[idx + self.seq_len - 1]  # 取序列的最後一天
        else:
            # 在 predict 模式下，生成未來 5 天的日期
            last_date = pd.to_datetime(self.dates[-1])
            date = [(last_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, self.pred_len + 1)]

        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        std = np.where(std == 0, 1, std)
        x = (x - mean) / std
        y = (y - mean) / std

        return {
            "x": torch.FloatTensor(x),
            "y": torch.FloatTensor(y),
            "date": date,
            "mean": torch.FloatTensor(mean),
            "std": torch.FloatTensor(std)
        }

    def get_enc_in(self):
        return self.enc_in