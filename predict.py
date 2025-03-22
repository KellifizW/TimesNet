import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from exp.exp_long_term_forecast import Exp_Long_Term_Forecast
from data_provider.data_loader import Dataset_Custom


class Args:
    def __init__(self):
        self.model = 'TimesNet'
        self.data = 'custom'
        self.root_path = 'data/raw'
        self.data_path = 'aapl_daily.csv'
        self.features = 'S'
        self.seq_len = 30
        self.pred_len = 5
        self.d_model = 64
        self.e_layers = 2
        self.top_k = 3
        self.batch_size = 32
        self.checkpoints = 'checkpoints'
        self.task_name = 'long_term_forecast'
        self.use_gpu = True


def predict():
    args = Args()

    # 載入測試數據
    test_dataset = Dataset_Custom(args.root_path, flag='test', size=[args.seq_len, args.pred_len],
                                  data_path=args.data_path)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 初始化實驗
    exp = Exp_Long_Term_Forecast(args)

    # 載入最佳模型
    exp.model.load_state_dict(torch.load(os.path.join(args.checkpoints, 'checkpoint.pth')))

    # 預測
    print("Predicting...")
    preds = []
    trues = []
    exp.model.eval()
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.float().cuda()
            batch_y = batch_y.float().cuda()
            outputs = exp.model(batch_x)
            preds.append(outputs.cpu().numpy())
            trues.append(batch_y.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    # 反正規化
    preds = test_dataset.inverse_transform(preds)
    trues = test_dataset.inverse_transform(trues)

    # 儲存預測結果
    os.makedirs('results', exist_ok=True)
    np.save('results/preds.npy', preds)
    np.save('results/trues.npy', trues)

    print("Prediction finished.")


if __name__ == "__main__":
    predict()