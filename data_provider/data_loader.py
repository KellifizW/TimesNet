import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class StockDataset(Dataset):
    def __init__(self, data_path, seq_len=15, pred_len=5, split="train", train_ratio=0.7, valid_ratio=0.2):
        df = pd.read_csv(data_path)
        self.data = df["Close"].values.astype(float)
        self.dates = df["Date"].values

        total_len = len(self.data)
        train_end = int(total_len * train_ratio)
        valid_end = int(total_len * (train_ratio + valid_ratio))

        if split == "train":
            self.data = self.data[:train_end]
            self.dates = self.dates[:train_end]
        elif split == "valid":
            self.data = self.data[train_end:valid_end]
            self.dates = self.dates[train_end:valid_end]
        else:  # test
            self.data = self.data[valid_end:]
            self.dates = self.dates[valid_end:]

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.mean = np.mean(self.data)
        self.std = np.std(self.data)
        self.data = (self.data - self.mean) / self.std

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len]
        date = self.dates[idx + self.seq_len - 1]

        return {
            "x": torch.FloatTensor(x),
            "y": torch.FloatTensor(y),
            "date": date,
            "mean": self.mean,
            "std": self.std
        }