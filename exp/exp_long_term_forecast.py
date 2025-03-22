import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import pandas as pd
from exp.exp_basic import Exp_Basic
from utils.metrics import MSE, MAE


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.train_losses = []
        self.val_losses = []

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_gpu:
            model = model.cuda()
        return model

    def _get_data(self, flag):
        from data_provider.data_factory import data_provider
        data_set = data_provider(self.args, flag)
        return data_set

    def _select_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def train(self, train_loader, val_loader):
        criterion = self._select_criterion()
        optimizer = self._select_optimizer()

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.args.train_epochs):
            self.model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                batch_x = batch_x.float().cuda()
                batch_y = batch_y.float().cuda()

                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)

            # 驗證
            val_loss = self.validate(val_loader, criterion)
            self.val_losses.append(val_loss)

            print(f"Epoch {epoch + 1}/{self.args.train_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), os.path.join(self.args.checkpoints, 'checkpoint.pth'))
            else:
                patience_counter += 1
                if patience_counter >= self.args.patience:
                    print("Early stopping triggered.")
                    break

        # 儲存損失
        losses_df = pd.DataFrame({
            'epoch': range(1, len(self.train_losses) + 1),
            'train_loss': self.train_losses,
            'val_loss': self.val_losses
        })
        losses_df.to_csv('results/losses.csv', index=False)

    def validate(self, val_loader, criterion):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.float().cuda()
                batch_y = batch_y.float().cuda()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        return val_loss / len(val_loader)