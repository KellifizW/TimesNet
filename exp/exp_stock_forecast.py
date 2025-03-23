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
        if self.args.use_gpu:
            model = model.cuda()
        return model

    def _get_data(self, flag):
        data_set = self.args.data_loader(
            data_path=self.args.data_path,
            seq_len=self.args.seq_len,
            pred_len=self.args.pred_len,
            split=flag
        )
        return data_set

    def train(self):
        train_data = self._get_data(flag="train")
        valid_data = self._get_data(flag="valid")

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=False)

        model = self._build_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate)
        criterion = nn.MSELoss()

        best_loss = float("inf")
        os.makedirs("checkpoints", exist_ok=True)

        for epoch in range(self.args.train_epochs):
            model.train()
            train_loss = 0
            for batch in train_loader:
                x = batch["x"].unsqueeze(-1).to(self.device)
                y = batch["y"].unsqueeze(-1).to(self.device)

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
                    x = batch["x"].unsqueeze(-1).to(self.device)
                    y = batch["y"].unsqueeze(-1).to(self.device)
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
                torch.save(model.state_dict(), os.path.join("checkpoints", "best_model.pth"))

        # 儲存損失數據
        loss_df = pd.DataFrame({
            "Epoch": range(1, self.args.train_epochs + 1),
            "Train_Loss": self.losses["train"],
            "Valid_Loss": self.losses["valid"]
        })
        os.makedirs("results", exist_ok=True)
        loss_df.to_csv("results/losses.csv", index=False)

    def predict(self):
        test_data = self._get_data(flag="test")
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

        model = self._build_model()
        model.load_state_dict(torch.load(os.path.join("checkpoints", "best_model.pth")))
        model.eval()

        preds, trues, dates = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                x = batch["x"].unsqueeze(-1).to(self.device)
                y = batch["y"].unsqueeze(-1).to(self.device)
                mean = batch["mean"].item()
                std = batch["std"].item()

                outputs = model(x, None, None, None)
                outputs = outputs.squeeze(-1).cpu().numpy() * std + mean
                y = y.squeeze(-1).cpu().numpy() * std + mean

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