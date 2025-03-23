import torch
import torch.nn as nn
from exp.exp_basic import Exp_Basic
import os
import numpy as np
import pandas as pd


class Exp_Stock_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Stock_Forecast, self).__init__(args)
        self.losses = {"train": [], "valid": []}

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        return model

    def _get_data(self, flag):
        data_set = self.args.data_loader(
            data_path=self.args.data_path,
            seq_len=self.args.seq_len,
            pred_len=self.args.pred_len,
            split=flag
        )
        return data_set

    def train(self, setting):
        train_data = self._get_data(flag="train")
        valid_data = self._get_data(flag="val")

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=False)

        model = self.model
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate, weight_decay=1e-5)
        criterion = nn.MSELoss()

        best_loss = float("inf")
        os.makedirs("checkpoints", exist_ok=True)

        for epoch in range(self.args.train_epochs):
            model.train()
            train_loss = 0
            for batch in train_loader:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)

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
                    x = batch["x"].to(self.device)
                    y = batch["y"].to(self.device)
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
                checkpoint_path = os.path.join("checkpoints", f"{setting}_best_model.pth")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")

        loss_df = pd.DataFrame({
            "Epoch": range(1, self.args.train_epochs + 1),
            "Train_Loss": self.losses["train"],
            "Valid_Loss": self.losses["valid"]
        })
        os.makedirs("results", exist_ok=True)
        loss_df.to_csv(os.path.join("results", f"{setting}_losses.csv"), index=False)

    def predict(self, setting=None):
        try:
            data = self._get_data(flag="predict")
            data_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)
            print(f"Data loader size: {len(data_loader)}")

            model = self.model
            if setting is None:
                raise ValueError("Setting must be provided to load the correct checkpoint file.")
            checkpoint_path = os.path.join("checkpoints", f"{setting}_best_model.pth")
            print(f"Loading checkpoint from {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
            model.eval()

            preds, dates = [], []
            with torch.no_grad():
                for batch in data_loader:
                    x = batch["x"].to(self.device)
                    mean = batch["mean"].to(self.device)
                    std = batch["std"].to(self.device)

                    outputs = model(x, None, None, None)
                    outputs = outputs * std + mean

                    outputs = outputs.cpu().numpy()
                    preds.append(outputs)
                    dates.append(batch["date"])  # date 是一個列表，例如 ["2025-03-24", ...]

            if not preds:
                raise ValueError("No predictions generated. Check data or model.")
            preds = np.concatenate(preds, axis=0)
            dates = np.array(dates)  # 形狀為 (1, pred_len)，例如 (1, 5)
            print(f"Prediction completed: {len(preds)} samples")

            os.makedirs("results", exist_ok=True)
            prefix = f"{setting}_"
            preds_path = os.path.join("results", f"{prefix}preds.npy")
            dates_path = os.path.join("results", f"{prefix}dates.npy")
            np.save(preds_path, preds)
            np.save(dates_path, dates)
            print(f"Prediction results saved to {preds_path}, {dates_path}")
        except Exception as e:
            print(f"Error in predict method: {str(e)}")
            raise

    def test(self, setting, test=0):
        test_data = self._get_data(flag="test")
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False, drop_last=False)

        model = self.model
        checkpoint_path = os.path.join("checkpoints", f"{setting}_best_model.pth")
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        model.eval()

        test_loss = 0
        criterion = nn.MSELoss()
        all_preds = []
        all_trues = []
        all_dates = []

        with torch.no_grad():
            for batch in test_loader:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)
                mean = batch["mean"].to(self.device)
                std = batch["std"].to(self.device)
                outputs = model(x, None, None, None)
                loss = criterion(outputs, y)
                test_loss += loss.item()

                # 調整 mean 和 std 的形狀以匹配 outputs
                mean = mean.unsqueeze(1).expand_as(outputs)
                std = std.unsqueeze(1).expand_as(outputs)

                # 反標準化
                outputs = outputs * std + mean
                y = y * std + mean
                all_preds.append(outputs.cpu().numpy())
                all_trues.append(y.cpu().numpy())
                all_dates.append(batch["date"])

        avg_test_loss = test_loss / len(test_loader)
        print(f"Test Loss: {avg_test_loss:.4f}")

        all_preds = np.concatenate(all_preds, axis=0)
        all_trues = np.concatenate(all_trues, axis=0)
        all_dates = np.concatenate(all_dates, axis=0)
        os.makedirs("results", exist_ok=True)
        np.save(f"results/{setting}_test_preds.npy", all_preds)
        np.save(f"results/{setting}_test_trues.npy", all_trues)
        np.save(f"results/{setting}_test_dates.npy", all_dates, allow_pickle=True)

        test_result_path = os.path.join("results", f"{setting}_test_result.txt")
        with open(test_result_path, 'w') as f:
            f.write(f"Test Loss: {avg_test_loss:.4f}\n")
        print(f"Test result saved to {test_result_path}")
        print(f"Test predictions saved to results/{setting}_test_preds.npy")