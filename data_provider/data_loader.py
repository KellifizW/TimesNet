import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class StockDataset(Dataset):
    def __init__(self, data_path, seq_len=15, pred_len=5, split="train", train_ratio=0.7, valid_ratio=0.2):
        df = pd.read_csv(data_path)
        # 使用所有特徵
        self.features = ['Open', 'High', 'Low', 'Close', 'Volume']
        self.data = df[self.features].values.astype(float)
        self.dates = df["Date"].values

        # 計算特徵數（enc_in）
        self.enc_in = self.data.shape[1]  # 特徵數，例如 5

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

        # 對每個特徵分別進行標準化
        self.mean = np.mean(self.data, axis=0)  # 形狀為 (enc_in,)
        self.std = np.std(self.data, axis=0)  # 形狀為 (enc_in,)
        # 避免除以 0
        self.std = np.where(self.std == 0, 1, self.std)
        self.data = (self.data - self.mean) / self.std

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]  # 形狀為 (seq_len, enc_in)
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len]  # 形狀為 (pred_len, enc_in)
        date = self.dates[idx + self.seq_len - 1]

        return {
            "x": torch.FloatTensor(x),  # 形狀為 (seq_len, enc_in)
            "y": torch.FloatTensor(y),  # 形狀為 (pred_len, enc_in)
            "date": date,
            "mean": torch.FloatTensor(self.mean),  # 形狀為 (enc_in,)
            "std": torch.FloatTensor(self.std)  # 形狀為 (enc_in,)
        }

    def get_enc_in(self):
        return self.enc_in