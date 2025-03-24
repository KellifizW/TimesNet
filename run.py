import os
import torch
import torch.backends
from exp.exp_stock_forecast import Exp_Stock_Forecast
from data_provider.data_loader import StockDataset
import random
import numpy as np
from visualize import visualize_forecast, visualize_loss, visualize_comparison, visualize_test_and_predict
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

def get_interactive_args():
    """交互式獲取用戶輸入的參數"""
    class Args:
        # 預設參數
        task_name = 'long_term_forecast'
        is_training = 1
        model_id = 'stock_forecast'
        model = 'TimesNet'
        seq_len = 15
        label_len = 0
        pred_len = 5
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
        num_workers = 0
        itr = 1
        train_epochs = 150
        batch_size = 32
        patience = 3
        learning_rate = 0.0001
        des = 'test'
        loss = 'MSE'
        lradj = 'type1'
        use_gpu = True
        gpu = 0
        gpu_type = 'cuda'  # cuda or mps
        use_multi_gpu = False
        devices = '0,1,2,3'
        data = 'stock'
        checkpoints = './checkpoints/'

    args = Args()

    # 交互式輸入股票代號
    ticker = input("請輸入股票代號（例如 AAPL，預設 AAPL）：").strip().upper() or "AAPL"
    print(f"正在抓取 {ticker} 的股票數據...")
    data_dir = "data/raw"
    from fetch_data import fetch_stock_data
    try:
        # 調用 fetch_stock_data，並獲取生成的數據文件路徑
        data_path = fetch_stock_data(ticker=ticker, output_dir=data_dir)
        if data_path is None:
            raise ValueError("fetch_stock_data 返回了 None，無法獲取數據文件路徑。")
        args.data_path = data_path
    except Exception as e:
        print(f"無法下載股票數據：{str(e)}")
        print("程式將退出。")
        exit(1)

    # 定義可用模型列表（與 exp_stock_forecast.py 中的 model_dict 保持一致）
    available_models = [
        'Autoformer', 'Crossformer', 'DLinear', 'FEDformer', 'FiLM', 'Informer',
        'iTransformer', 'Koopa', 'LightTS', 'MambaSimple', 'MICN', 'MultiPatchformer',
        'PatchTST', 'Reformer', 'TemporalFusionTransformer', 'TimeMixer', 'TimesNet',
        'Timexer', 'Transformer'
    ]

    # 交互式輸入關鍵參數
    print("\n請輸入參數（按 Enter 接受預設值）：")
    print(f"可用模型: {available_models}")
    while True:
        model_input = input(f"模型名稱（預設: {args.model}）：") or args.model
        if model_input in available_models:
            args.model = model_input
            break
        else:
            print(f"錯誤：'{model_input}' 不在可用模型中，請重新輸入。")

    args.seq_len = int(input(f"輸入序列長度（預設: {args.seq_len}）：") or args.seq_len)
    args.pred_len = int(input(f"預測長度（預設: {args.pred_len}）：") or args.pred_len)
    args.d_model = int(input(f"模型維度（預設: {args.d_model}）：") or args.d_model)
    args.d_ff = int(input(f"前饋層維度（預設: {args.d_ff}）：") or args.d_ff)
    args.e_layers = int(input(f"編碼器層數（預設: {args.e_layers}）：") or args.e_layers)
    args.dropout = float(input(f"Dropout 比例（預設: {args.dropout}）：") or args.dropout)
    args.learning_rate = float(input(f"學習率（預設: {args.learning_rate}）：") or args.learning_rate)
    args.train_epochs = int(input(f"訓練週期（預設: {args.train_epochs}）：") or args.train_epochs)
    args.batch_size = int(input(f"批次大小（預設: {args.batch_size}）：") or args.batch_size)

    return args

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
        visualize_test_and_predict(setting=setting, feature_idx=3)  # 新增：測試集與預測連接圖

    elif choice == 2:  # 僅預測
        print('>>>>>>>預測 : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.predict(setting)

        print('>>>>>>>視覺化 : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        visualize_forecast(setting=setting, feature_idx=3)  # 輸出未來 5 天預測結果
        visualize_comparison(setting=setting, feature_idx=3, historical_data_path=args.data_path)
        visualize_test_and_predict(setting=setting, feature_idx=3)  # 新增：僅預測時也顯示

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
        args = get_interactive_args()  # 獲取交互式參數
        run_experiment(args, choice)
        print("\n執行完成！")