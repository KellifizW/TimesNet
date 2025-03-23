import argparse
import torch
from exp.exp_stock_forecast import Exp_Stock_Forecast
from data_provider.data_loader import StockDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="TimesNet")
    parser.add_argument("--data_path", type=str, default="data/raw/aapl_daily.csv")
    parser.add_argument("--seq_len", type=int, default=15)
    parser.add_argument("--pred_len", type=int, default=5)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--d_model", type=int, default=90)
    parser.add_argument("--d_ff", type=int, default=90)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--num_kernels", type=int, default=6)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gpu_type", type=str, default="cuda")  # 添加 gpu_type
    parser.add_argument("--label_len", type=int, default=0)  # 添加 label_len
    parser.add_argument("--embed", type=str, default="timeF")  # 添加 embed
    parser.add_argument("--freq", type=str, default="d")  # 添加 freq
    args = parser.parse_args()

    # 動態設置 enc_in, dec_in, c_out
    dataset = StockDataset(data_path=args.data_path, seq_len=args.seq_len, pred_len=args.pred_len, split="test")
    args.enc_in = dataset.get_enc_in()  # 動態獲取 enc_in，例如 5
    args.dec_in = args.enc_in  # 解碼器輸入維度與編碼器相同
    args.c_out = args.enc_in  # 輸出維度與編碼器相同（預測所有特徵）
    # 如果只預測 Close，可以設置：
    # args.c_out = 1

    # 設置設備
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:0')
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')

    args.data_loader = StockDataset
    exp = Exp_Stock_Forecast(args)
    exp.predict()

    # 清理 GPU 記憶體
    if args.gpu_type == 'mps':
        torch.backends.mps.empty_cache()
    elif args.gpu_type == 'cuda':
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()