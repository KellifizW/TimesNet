import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

class StockDataset(Dataset):
    def __init__(self, root_path, data_path, flag='train', size=None, features='MS', target='Close', timeenc=1, freq='d'):
        # 設置序列長度參數
        if size is None:
            self.seq_len = 15
            self.label_len = 10
            self.pred_len = 5
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        self.flag = flag
        self.features = features
        self.target = target
        self.timeenc = timeenc
        self.freq = freq

        # 加載數據
        self.data_path = os.path.join(root_path, data_path)
        df_raw = pd.read_csv(self.data_path)
        self.cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        self.data = df_raw[self.cols].values.astype(float)
        self.dates = df_raw["Date"].values  # 保存所有日期

        self.enc_in = self.data.shape[1]

        # 分片數據
        total_len = len(self.data)
        num_train = int(total_len * 0.7)
        num_vali = int(total_len * 0.1)
        num_test = total_len - num_train - num_vali

        border1s = [0, num_train - self.seq_len, total_len - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, total_len]
        type_map = {"train": 0, "val": 1, "test": 2, "predict": 3}
        self.set_type = type_map[flag]

        if flag == "predict":
            border1 = total_len - self.seq_len
            border2 = total_len
        else:
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

        # 檢查邊界條件，避免越界
        if border1 < 0 or border2 > total_len or border1 >= border2:
            raise ValueError(f"Invalid border range for {flag}: [{border1}, {border2}] with total length {total_len}")

        # 全局標準化
        self.scaler = StandardScaler()
        self.scale = True
        train_data = self.data[border1s[0]:border2s[0]]
        self.scaler.fit(train_data)
        self.data = self.scaler.transform(self.data)

        # 保存均值和標準差（用於反標準化）
        self.means = self.scaler.mean_
        self.stds = np.sqrt(self.scaler.var_)
        print(f"Scaler means: {self.means}")
        print(f"Scaler stds: {self.stds}")

        # 提取時間特徵
        df_stamp = pd.DataFrame({"date": self.dates})
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday())
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            from utils.timefeatures import time_features
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        # 根據 split 選擇數據範圍
        self.data_x = self.data[border1:border2]
        self.data_y = self.data[border1:border2]
        self.data_stamp = data_stamp[border1:border2]
        self.dates = self.dates[border1:border2]  # 保存切分後的日期

        # 檢查數據長度是否足以生成序列
        if len(self.data_x) < self.seq_len + self.pred_len and flag != "predict":
            raise ValueError(f"Data length {len(self.data_x)} too short for seq_len={self.seq_len} and pred_len={self.pred_len}")

    def __len__(self):
        if self.flag == "predict":
            return 1
        return max(1, len(self.data_x) - self.seq_len - self.pred_len + 1)  # 確保至少返回 1

    def __getitem__(self, idx):
        if self.flag == "predict":
            idx = 0

        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # 檢查索引範圍
        if s_end > len(self.data_x) or r_end > len(self.data_y):
            raise IndexError(f"Index out of range: s_end={s_end}, r_end={r_end}, data_len={len(self.data_x)}")

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        # 返回元組，與原始 train 方法兼容
        return (torch.FloatTensor(seq_x), torch.FloatTensor(seq_y),
                torch.FloatTensor(seq_x_mark), torch.FloatTensor(seq_y_mark))

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def get_enc_in(self):
        return self.enc_in

    def get_dates(self):
        return self.dates

    def get_mean(self):
        return self.means

    def get_std(self):
        return self.stds