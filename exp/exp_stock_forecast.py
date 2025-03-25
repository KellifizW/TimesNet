import torch
import torch.nn as nn
from exp.exp_basic import Exp_Basic
import os
import time
import numpy as np
import pandas as pd
from data_utils import generate_trading_days

# 導入所有模型
from models import (
    Autoformer, Crossformer, DLinear, FEDformer, FiLM, Informer, iTransformer,
    Koopa, LightTS, MambaSimple, MICN, MultiPatchFormer, PatchTST, Reformer,
    TemporalFusionTransformer, TimeMixer, TimesNet, TimeXer, Transformer
)

from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.dtw_metric import dtw, accelerated_dtw

class Exp_Stock_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Stock_Forecast, self).__init__(args)
        self.losses = {"train": [], "valid": []}
        self.model_dict = {
            'Autoformer': Autoformer,
            'Crossformer': Crossformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'FiLM': FiLM,
            'Informer': Informer,
            'iTransformer': iTransformer,
            'Koopa': Koopa,
            'LightTS': LightTS,
            'MambaSimple': MambaSimple,
            'MICN': MICN,
            'MultiPatchformer': MultiPatchFormer,
            'PatchTST': PatchTST,
            'Reformer': Reformer,
            'TemporalFusionTransformer': TemporalFusionTransformer,
            'TimeMixer': TimeMixer,
            'TimesNet': TimesNet,
            'Timexer': TimeXer,
            'Transformer': Transformer,
        }

    def _build_model(self):
        if self.args.model not in self.model_dict:
            raise ValueError(f"Model '{self.args.model}' not found. Available models: {list(self.model_dict.keys())}")
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        from data_provider.data_factory import data_provider
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        # 只取收盤價的預測 (假設收盤價在第 3 列)
                        outputs = outputs[:, :, 3:4]  # 調整為 (batch_size, pred_len, 1)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    # 只取收盤價的預測
                    outputs = outputs[:, :, 3:4]  # 調整為 (batch_size, pred_len, 1)

                batch_y = batch_y[:, -self.args.pred_len:, 3:4].to(self.device)
                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        outputs = outputs[:, :, 3:4]  # 只取收盤價
                        batch_y = batch_y[:, -self.args.pred_len:, 3:4].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs = outputs[:, :, 3:4]  # 只取收盤價
                    batch_y = batch_y[:, -self.args.pred_len:, 3:4].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            self.losses["train"].append(train_loss)
            self.losses["valid"].append(vali_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path, weights_only=True))

        loss_df = pd.DataFrame({
            "Epoch": range(1, len(self.losses["train"]) + 1),
            "Train_Loss": self.losses["train"],
            "Valid_Loss": self.losses["valid"]
        })
        os.makedirs("results", exist_ok=True)
        loss_df.to_csv(os.path.join("results", f"{setting}_losses.csv"), index=False)

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="test")
        if test:
            print('loading model')
            checkpoint_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        outputs = outputs[:, :, 3:4]  # 只取收盤價
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs = outputs[:, :, 3:4]  # 只取收盤價

                batch_y = batch_y[:, -self.args.pred_len:, 3:4].to(self.device)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs_flat = outputs.reshape(-1, 1)
                    batch_y_flat = batch_y.reshape(-1, 1)
                    close_mean = test_data.get_mean()[3]
                    close_std = test_data.get_std()[3]
                    outputs = outputs_flat * close_std + close_mean
                    batch_y = batch_y_flat * close_std + close_mean
                    outputs = outputs.reshape(shape)
                    batch_y = batch_y.reshape(shape)

                preds.append(outputs)
                trues.append(batch_y)

                if i % 20 == 0 and batch_y.shape[0] > 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input_flat = input.reshape(-1, input.shape[-1])
                        input = test_data.inverse_transform(input_flat).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], batch_y[0, :, 0]), axis=0)
                    pred_data = np.concatenate((input[0, :, -1], outputs[0, :, 0]), axis=0)
                    visual(gt, pred_data, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0).squeeze(-1)
        trues = np.concatenate(trues, axis=0).squeeze(-1)
        print('test shape:', preds.shape, trues.shape)

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        with open("result_stock_forecast.txt", 'a') as f:
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
            f.write('\n')
            f.write('\n')

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        np.save(f"results/{setting}_test_preds.npy", preds)
        np.save(f"results/{setting}_test_trues.npy", trues)

    def predict(self, setting=None):
        data, data_loader = self._get_data(flag="predict")
        checkpoint_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        self.model.eval()

        preds = []
        predict_dates = data.get_dates()
        last_date = predict_dates[-1]
        future_dates = generate_trading_days(last_date, self.args.pred_len)

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        outputs = outputs[:, :, 3:4]  # 只取收盤價
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs = outputs[:, :, 3:4]  # 只取收盤價

                outputs = outputs.detach().cpu().numpy()

                if data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs_flat = outputs.reshape(-1, 1)
                    close_mean = data.get_mean()[3]
                    close_std = data.get_std()[3]
                    outputs_restored = outputs_flat * close_std + close_mean
                    outputs_restored = outputs_restored.reshape(shape)
                    close_outputs_restored = outputs_restored
                else:
                    close_outputs_restored = outputs

                preds.append(close_outputs_restored)

        preds = np.concatenate(preds, axis=0).squeeze(-1)
        dates = np.array([future_dates] * preds.shape[0])
        return preds, dates

# 主程式執行範例（根據你的參數）
if __name__ == "__main__":
    from argparse import Namespace
    import torch

    args = Namespace(
        task_name='long_term_forecast',
        is_training=1,
        model_id='stock_forecast',
        model='TimesNet',
        features='MS',
        data='stock',
        root_path='data/raw',
        data_path='aapl_daily.csv',
        years=3,
        seq_len=15,
        label_len=10,
        pred_len=5,
        freq='d',
        timeenc=1,
        checkpoints='./checkpoints/',
        inverse=True,
        top_k=5,
        num_kernels=6,
        enc_in=5,
        dec_in=5,
        c_out=1,
        d_model=90,
        d_ff=90,
        e_layers=2,
        dropout=0.1,
        embed='timeF',
        num_workers=0,
        itr=1,
        train_epochs=150,
        batch_size=32,
        patience=3,
        learning_rate=0.0001,
        des='test',
        loss='MSE',
        lradj='type1',
        use_amp=False,
        use_dtw=False,
        use_gpu=True,
        gpu=0,
        gpu_type='cuda',
        use_multi_gpu=False,
        devices='0,1,2,3'
    )
    args.device = torch.device('cuda:0' if args.use_gpu and torch.cuda.is_available() else 'cpu')

    setting = f"{args.task_name}_{args.model_id}_{args.model}_sl{args.seq_len}_pl{args.pred_len}_dm{args.d_model}_df{args.d_ff}_el{args.e_layers}_dropout{args.dropout}_{args.des}"
    exp = Exp_Stock_Forecast(args)
    print(f">>>>>>>開始訓練 : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
    exp.train(setting)
    print(f">>>>>>>測試 : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    exp.test(setting)