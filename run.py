import os
import torch
import torch.backends
from exp.exp_stock_forecast import Exp_Stock_Forecast
from data_provider.data_loader import StockDataset
import random
import numpy as np
from visualize import visualize_forecast, visualize_loss, visualize_comparison
from backtest import backtest  # 導入回測函數


def get_user_choice():
    """顯示選項並獲取用戶輸入"""
    print("\n請選擇要執行的程序：")
    print("1. 訓練並預測 (Train and Predict)")
    print("2. 僅預測 (Predict Only)")
    print("3. 退出 (Exit)")

    while True:
        try:
            choice = int(input("輸入您的選擇 (1-3): "))
            if choice in [1, 2, 3]:
                return choice
            else:
                print("無效選擇，請輸入 1、2 或 3。")
        except ValueError:
            print("請輸入有效的數字！")


def run_experiment(args, choice):
    """根據用戶選擇執行實驗"""
    # 動態設置 enc_in, dec_in, c_out
    dataset = StockDataset(data_path=args.data_path, seq_len=args.seq_len, pred_len=args.pred_len, split="train")
    args.enc_in = dataset.get_enc_in()  # 動態獲取 enc_in，例如 5
    args.dec_in = args.enc_in  # 解碼器輸入維度與編碼器相同
    args.c_out = args.enc_in  # 輸出維度與編碼器相同
    args.data_loader = StockDataset

    # 設置設備
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('使用 GPU')
    else:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.device = torch.device("mps")
            print('使用 MPS')
        else:
            args.device = torch.device("cpu")
            print('使用 CPU')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('實驗參數:')
    print(args.__dict__)  # 顯示所有參數

    Exp = Exp_Stock_Forecast
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

    # 根據選擇執行模式
    if choice == 1:  # 訓練並預測
        if args.is_training:
            print('>>>>>>>開始訓練 : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>測試 : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            # 執行回測
            print('>>>>>>>回測 : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            test_data = exp._get_data(flag="test")
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
            backtest(setting, exp.model, test_loader, exp.device, args.seq_len, args.pred_len, feature_idx=3)

        print('>>>>>>>預測 : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.predict(setting)

        print('>>>>>>>視覺化 : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        visualize_loss(setting=setting)  # 繪製損失曲線圖
        visualize_forecast(setting=setting, feature_idx=3)  # 輸出未來 5 天預測結果
        visualize_comparison(setting=setting, feature_idx=3, historical_data_path=args.data_path)

    elif choice == 2:  # 僅預測
        print('>>>>>>>預測 : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.predict(setting)

        print('>>>>>>>視覺化 : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        visualize_forecast(setting=setting, feature_idx=3)  # 輸出未來 5 天預測結果
        visualize_comparison(setting=setting, feature_idx=3, historical_data_path=args.data_path)

    # 清理 GPU 記憶體
    if args.gpu_type == 'mps' and hasattr(torch.backends, "mps"):
        torch.backends.mps.empty_cache()
    elif args.gpu_type == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    # 固定隨機種子
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)


    # 設置默認參數（使用類似 argparse 的方式，但不依賴命令行）
    class Args:
        # basic config
        task_name = 'long_term_forecast'
        is_training = 1
        model_id = 'stock_forecast'
        model = 'TimesNet'

        # data loader
        data = 'stock'
        data_path = 'data/raw/aapl_daily.csv'
        checkpoints = './checkpoints/'

        # forecasting task
        seq_len = 15
        label_len = 0
        pred_len = 5

        # model define
        top_k = 5
        num_kernels = 6
        enc_in = 1  # 將動態設置
        dec_in = 1  # 將動態設置
        c_out = 1  # 將動態設置
        d_model = 90
        d_ff = 90
        e_layers = 2
        dropout = 0.1
        embed = 'timeF'
        freq = 'd'

        # optimization
        num_workers = 0
        itr = 1
        train_epochs = 150
        batch_size = 32
        patience = 3
        learning_rate = 0.0001
        des = 'test'
        loss = 'MSE'
        lradj = 'type1'

        # GPU
        use_gpu = True
        gpu = 0
        gpu_type = 'cuda'  # cuda or mps
        use_multi_gpu = False
        devices = '0,1,2,3'


    args = Args()

    # 主循環：持續詢問用戶選擇
    while True:
        choice = get_user_choice()
        if choice == 3:  # 退出
            print("程序結束。")
            break
        run_experiment(args, choice)
        print("\n執行完成！")