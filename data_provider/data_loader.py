import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='aapl_daily.csv', scale=True):
        self.seq_len = size[0]  # 序列長度，例如30
        self.pred_len = size[1]  # 預測長度，例如5

        # 載入數據
        self.data_path = data_path
        self.root_path = root_path
        self.flag = flag
        self.scale = scale

        # 讀取CSV檔案
        df_raw = pd.read_csv(f"{root_path}/{data_path}")
        self.data = df_raw['close'].values
        self.dates = pd.to_datetime(df_raw['date']).values

        # 數據分割：70%訓練，20%驗證，10%測試
        num_train = int(len(self.data) * 0.7)
        num_val = int(len(self.data) * 0.2)
        num_test = len(self.data) - num_train - num_val

        if flag == 'train':
            self.start_idx = 0
            self.end_idx = num_train
        elif flag == 'val':
            self.start_idx = num_train
            self.end_idx = num_train + num_val
        else:  # test
            self.start_idx = num_train + num_val
            self.end_idx = len(self.data)

        # 正規化
        if self.scale:
            self.scaler = StandardScaler()
            train_data = self.data[:num_train]
            self.scaler.fit(train_data.reshape(-1, 1))
            self.data = self.scaler.transform(self.data.reshape(-1, 1)).flatten()

    def __getitem__(self, index):
        s_begin = self.start_idx + index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]

        return seq_x, seq_y

    def __len__(self):
        return self.end_idx - self.start_idx - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        if self.scale:
            return self.scaler.inverse_transform(data.reshape(-1, 1)).flatten()
        return data

    def get_dates(self, index):
        s_begin = self.start_idx + index
        s_end = s_begin + self.seq_len + self.pred_len
        return self.dates[s_begin:s_end]


class StandardScaler:
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = data.mean()
        self.std = data.std()

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean