import torch
from torch.utils.data import DataLoader
import os
from exp.exp_long_term_forecast import Exp_Long_Term_Forecast
from data_provider.data_loader import Dataset_Custom


class Args:
    def __init__(self):
        self.model = 'TimesNet'
        self.data = 'custom'
        self.root_path = 'data/raw'
        self.data_path = 'aapl_daily.csv'
        self.features = 'S'  # 單變量
        self.seq_len = 30
        self.pred_len = 5
        self.d_model = 64
        self.e_layers = 2
        self.top_k = 3
        self.batch_size = 32
        self.learning_rate = 0.001
        self.num_workers = 0
        self.train_epochs = 10
        self.patience = 3
        self.checkpoints = 'checkpoints'
        self.task_name = 'long_term_forecast'
        self.use_gpu = True


def train():
    args = Args()

    # 創建檢查點目錄
    os.makedirs(args.checkpoints, exist_ok=True)

    # 載入數據
    train_dataset = Dataset_Custom(args.root_path, flag='train', size=[args.seq_len, args.pred_len],
                                   data_path=args.data_path)
    val_dataset = Dataset_Custom(args.root_path, flag='val', size=[args.seq_len, args.pred_len],
                                 data_path=args.data_path)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 初始化實驗
    exp = Exp_Long_Term_Forecast(args)

    # 訓練模型
    print("Training...")
    exp.train(train_loader, val_loader)

    print("Training finished.")


if __name__ == "__main__":
    train()