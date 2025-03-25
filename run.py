import os
import torch
import torch.backends
from exp.exp_stock_forecast import Exp_Stock_Forecast
import random
import numpy as np
from visualize import visualize_forecast, visualize_loss, visualize_comparison, visualize_test_and_predict
from backtest import backtest
import argparse

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

def get_args():
    """使用 argparse 解析命令行參數，並結合交互式輸入"""
    parser = argparse.ArgumentParser(description='Stock Forecasting with TimesNet')

    # 基本參數
    parser.add_argument('--task_name', type=str, default='long_term_forecast', help='task name')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='stock_forecast', help='model id')
    parser.add_argument('--model', type=str, default='TimesNet', help='model name')
    parser.add_argument('--features', type=str, default='MS',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')

    # 數據參數
    parser.add_argument('--data', type=str, default='stock', help='dataset type')
    parser.add_argument('--root_path', type=str, default='data/raw/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default=None, help='data file path')
    parser.add_argument('--years', type=int, default=5, help='number of years of stock data to download')
    parser.add_argument('--seq_len', type=int, default=15, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=10, help='start token length')
    parser.add_argument('--pred_len', type=int, default=5, help='prediction sequence length')
    parser.add_argument('--freq', type=str, default='d', help='freq for time features encoding')
    parser.add_argument('--timeenc', type=int, default=1, help='time encoding type: 0 for manual, 1 for time_features')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # 模型參數
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=90, help='dimension of model')
    parser.add_argument('--d_ff', type=int, default=90, help='dimension of fcn')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')

    # 優化參數
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=150, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--use_dtw', type=bool, default=False, help='use dtw metric')

    # GPU 參數
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multiple gpus')

    args = parser.parse_args()

    # 交互式輸入：如果某些參數未通過命令行指定，則進入交互模式
    if args.data_path is None:
        ticker = input("請輸入股票代號（例如 AAPL，預設 AAPL）：").strip().upper() or "AAPL"
        args.years = int(input(f"請輸入下載數據的年份（預設: {args.years}）：") or args.years)
        print(f"正在抓取 {ticker} 的股票數據（{args.years} 年）...")
        data_dir = "data/raw"
        from fetch_data import fetch_stock_data
        try:
            data_path = fetch_stock_data(ticker=ticker, years=args.years, output_dir=data_dir)
            if data_path is None:
                raise ValueError("fetch_stock_data 返回了 None，無法獲取數據文件路徑。")
            args.data_path = os.path.basename(data_path)  # 只需要文件名，例如 aapl_daily.csv
            args.root_path = data_dir  # 根目錄，例如 data/raw/
        except Exception as e:
            print(f"無法下載股票數據：{str(e)}")
            print("程式將退出。")
            exit(1)

    # 定義可用模型列表
    available_models = [
        'Autoformer', 'Crossformer', 'DLinear', 'FEDformer', 'FiLM', 'Informer',
        'iTransformer', 'Koopa', 'LightTS', 'MambaSimple', 'MICN', 'MultiPatchformer',
        'PatchTST', 'Reformer', 'TemporalFusionTransformer', 'TimeMixer', 'TimesNet',
        'Timexer', 'Transformer'
    ]

    # 交互式輸入其他參數（如果命令行未指定）
    print("\n請輸入參數（按 Enter 接受預設值或命令行指定的值）：")
    print(f"可用模型: {available_models}")
    while True:
        model_input = input(f"模型名稱（預設: {args.model}）：") or args.model
        if model_input in available_models:
            args.model = model_input
            break
        else:
            print(f"錯誤：'{model_input}' 不在可用模型中，請重新輸入。")

    args.seq_len = int(input(f"輸入序列長度（預設: {args.seq_len}）：") or args.seq_len)
    args.label_len = int(input(f"標籤序列長度（預設: {args.label_len}）：") or args.label_len)
    args.pred_len = int(input(f"預測長度（預設: {args.pred_len}）：") or args.pred_len)
    args.d_model = int(input(f"模型維度（預設: {args.d_model}）：") or args.d_model)
    args.d_ff = int(input(f"前饋層維度（預設: {args.d_ff}）：") or args.d_ff)
    args.e_layers = int(input(f"編碼器層數（預設: {args.e_layers}）：") or args.e_layers)
    args.dropout = float(input(f"Dropout 比例（預設: {args.dropout}）：") or args.dropout)
    args.learning_rate = float(input(f"學習率（預設: {args.learning_rate}）：") or args.learning_rate)
    args.train_epochs = int(input(f"訓練週期（預設: {args.train_epochs}）：") or args.train_epochs)
    args.batch_size = int(input(f"批次大小（預設: {args.batch_size}）：") or args.batch_size)
    args.timeenc = int(input(f"時間特徵編碼方式 (0: 手動, 1: time_features, 預設: {args.timeenc})：") or args.timeenc)

    return args

def run_experiment(args, choice):
    """根據用戶選擇執行實驗"""
    # 動態設置 enc_in, dec_in, c_out
    args.inverse = True
    from data_provider.data_factory import data_provider
    dataset, _ = data_provider(args, "train")
    args.enc_in = dataset.get_enc_in()
    args.dec_in = args.enc_in
    args.c_out = 1 if args.features == 'MS' else args.enc_in

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
    print(vars(args))

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
            test_data, _ = exp._get_data(flag="test")  # 修改：只取 data_set，忽略 data_loader
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
            backtest(setting, exp.model, test_loader, exp.device, args.seq_len, args.pred_len, feature_idx=3)

        print('>>>>>>>預測 : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.predict(setting)

        print('>>>>>>>視覺化 : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        visualize_loss(setting=setting)
        visualize_forecast(setting=setting, feature_idx=3)
        visualize_comparison(setting=setting, feature_idx=3, historical_data_path=os.path.join(args.root_path, args.data_path))
        visualize_test_and_predict(setting=setting, feature_idx=3, pred_len=args.pred_len)  # 傳入 pred_len

    elif choice == 2:  # 僅預測
        print('>>>>>>>預測 : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.predict(setting)

        print('>>>>>>>視覺化 : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        visualize_forecast(setting=setting, feature_idx=3)
        visualize_comparison(setting=setting, feature_idx=3, historical_data_path=os.path.join(args.root_path, args.data_path))
        visualize_test_and_predict(setting=setting, feature_idx=3, pred_len=args.pred_len)  # 傳入 pred_len

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

    # 主循環：持續詢問用戶選擇
    while True:
        choice = get_user_choice()
        if choice == 3:  # 退出
            print("程序結束。")
            break
        args = get_args()
        run_experiment(args, choice)
        print("\n執行完成！")