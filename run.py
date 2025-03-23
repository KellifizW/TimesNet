import argparse
import os
import torch
import torch.backends
from exp.exp_stock_forecast import Exp_Stock_Forecast
from data_provider.data_loader import StockDataset
import random
import numpy as np
from visualize import visualize_forecast, visualize_loss  # 導入繪圖函數

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='stock_forecast', help='model id')
    parser.add_argument('--model', type=str, default='TimesNet',
                        help='model name, options: [TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, default='stock', help='dataset type')
    parser.add_argument('--data_path', type=str, default='data/raw/aapl_daily.csv', help='data file')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=15, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--pred_len', type=int, default=5, help='prediction sequence length')

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')  # 臨時設置，稍後會覆蓋
    parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')  # 臨時設置，稍後會覆蓋
    parser.add_argument('--c_out', type=int, default=1, help='output size')  # 臨時設置，稍後會覆蓋
    parser.add_argument('--d_model', type=int, default=90, help='dimension of model')
    parser.add_argument('--d_ff', type=int, default=90, help='dimension of fcn')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--freq', type=str, default='d', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=150, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    args = parser.parse_args()

    # 動態設置 enc_in, dec_in, c_out
    dataset = StockDataset(data_path=args.data_path, seq_len=args.seq_len, pred_len=args.pred_len, split="train")
    args.enc_in = dataset.get_enc_in()  # 動態獲取 enc_in，例如 5
    args.dec_in = args.enc_in  # 解碼器輸入維度與編碼器相同
    args.c_out = args.enc_in  # 輸出維度與編碼器相同
    args.data_loader = StockDataset

    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')

    # 以下程式碼保持不變
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = Exp_Stock_Forecast

    if args.is_training:
        for ii in range(args.itr):
            exp = Exp(args)
            setting = '{}_{}_{}_sl{}_pl{}_dm{}_df{}_el{}_dropout{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.seq_len,
                args.pred_len,
                args.d_model,
                args.d_ff,
                args.e_layers,
                args.dropout,
                args.des)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting)

            # 繪製圖表
            print('>>>>>>>visualizing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            visualize_loss(setting=setting)  # 繪製損失曲線圖
            visualize_forecast(setting=setting, feature_idx=3)  # 繪製預測結果圖（僅繪製 Close）

            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
    else:
        exp = Exp(args)
        setting = '{}_{}_{}_sl{}_pl{}_dm{}_df{}_el{}_dropout{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.seq_len,
            args.pred_len,
            args.d_model,
            args.d_ff,
            args.e_layers,
            args.dropout,
            args.des)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        if args.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif args.gpu_type == 'cuda':
            torch.cuda.empty_cache()