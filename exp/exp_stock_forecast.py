import torch
import torch.nn as nn
from exp.exp_basic import Exp_Basic
import os
import numpy as np
import pandas as pd


class Exp_Stock_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Stock_Forecast, self).__init__(args)
        self.losses = {"train": [], "valid": []}  # 記錄損失

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        # 移除硬編碼的 model.cuda()，因為 Exp_Basic.__init__ 已將模型移動到 self.device
        return model

    def _get_data(self, flag):
        data_set = self.args.data_loader(
            data_path=self.args.data_path,
            seq_len=self.args.seq_len,
            pred_len=self.args.pred_len,
            split=flag
        )
        return data_set

    def train(self, setting):  # 添加 setting 參數
        train_data = self._get_data(flag="train")
        valid_data = self._get_data(flag="valid")

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=False)

        model = self.model
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate)
        criterion = nn.MSELoss()

        best_loss = float("inf")
        os.makedirs("checkpoints", exist_ok=True)

        for epoch in range(self.args.train_epochs):
            model.train()
            train_loss = 0
            for batch in train_loader:
                x = batch["x"].to(self.device)  # 形狀為 (batch_size, seq_len, enc_in)，移除 unsqueeze
                y = batch["y"].to(self.device)  # 形狀為 (batch_size, pred_len, c_out)，移除 unsqueeze

                optimizer.zero_grad()
                outputs = model(x, None, None, None)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            valid_loss = 0
            with torch.no_grad():
                for batch in valid_loader:
                    x = batch["x"].to(self.device)  # 形狀為 (batch_size, seq_len, enc_in)
                    y = batch["y"].to(self.device)  # 形狀為 (batch_size, pred_len, c_out)
                    outputs = model(x, None, None, None)
                    loss = criterion(outputs, y)
                    valid_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            avg_valid_loss = valid_loss / len(valid_loader)
            self.losses["train"].append(avg_train_loss)
            self.losses["valid"].append(avg_valid_loss)

            print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}")

            if avg_valid_loss < best_loss:
                best_loss = avg_valid_loss
                # 使用 setting 參數保存檢查點
                checkpoint_path = os.path.join("checkpoints", f"{setting}_best_model.pth")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")

        # 儲存損失數據
        loss_df = pd.DataFrame({
            "Epoch": range(1, self.args.train_epochs + 1),
            "Train_Loss": self.losses["train"],
            "Valid_Loss": self.losses["valid"]
        })
        os.makedirs("results", exist_ok=True)
        loss_df.to_csv(os.path.join("results", f"{setting}_losses.csv"), index=False)

    def predict(self, setting=None):
        test_data = self._get_data(flag="test")
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

        model = self.model
        # 使用 setting 參數加載檢查點（假設與 run.py 中的 setting 一致）
        # 注意：這裡需要確保 checkpoint 文件名與 train 方法中的一致
        # 由於 predict.py 可能不傳入 setting，這裡使用固定的文件名
        checkpoint_path = os.path.join("checkpoints", "best_model.pth")
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()

        preds, trues, dates = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                x = batch["x"].to(self.device)  # 形狀為 (batch_size, seq_len, enc_in)
                y = batch["y"].to(self.device)  # 形狀為 (batch_size, pred_len, c_out)
                mean = batch["mean"].to(self.device)  # 形狀為 (enc_in,)
                std = batch["std"].to(self.device)  # 形狀為 (enc_in,)

                outputs = model(x, None, None, None)
                # 反標準化：outputs 和 y 的形狀為 (batch_size, pred_len, c_out)
                outputs = outputs * std + mean
                y = y * std + mean

                outputs = outputs.cpu().numpy()
                y = y.cpu().numpy()

                preds.append(outputs)
                trues.append(y)
                dates.append(batch["date"])

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        dates = np.concatenate(dates, axis=0)

        os.makedirs("results", exist_ok=True)
        np.save("results/preds.npy", preds)
        np.save("results/trues.npy", trues)
        np.save("results/dates.npy", dates)

    # 添加 test 方法以符合 run.py 的調用
    def test(self, setting, test=0):
        test_data = self._get_data(flag="test")
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

        model = self.model
        checkpoint_path = os.path.join("checkpoints", f"{setting}_best_model.pth")
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        model.eval()

        test_loss = 0
        criterion = nn.MSELoss()
        with torch.no_grad():
            for batch in test_loader:
                x = batch["x"].to(self.device)  # 形狀為 (batch_size, seq_len, enc_in)
                y = batch["y"].to(self.device)  # 形狀為 (batch_size, pred_len, c_out)
                outputs = model(x, None, None, None)
                loss = criterion(outputs, y)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
        print(f"Test Loss: {avg_test_loss:.4f}")

        # 儲存測試結果
        test_result_path = os.path.join("results", f"{setting}_test_result.txt")
        with open(test_result_path, 'w') as f:
            f.write(f"Test Loss: {avg_test_loss:.4f}\n")
        print(f"Test result saved to {test_result_path}")