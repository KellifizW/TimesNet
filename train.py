import argparse
from exp.exp_stock_forecast import Exp_Stock_Forecast
from data_provider.data_loader import StockDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="TimesNet")
    parser.add_argument("--data_path", type=str, default="data/raw/aapl_daily.csv")
    parser.add_argument("--seq_len", type=int, default=15)
    parser.add_argument("--pred_len", type=int, default=5)
    parser.add_argument("--train_epochs", type=int, default=150)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--d_model", type=int, default=90)
    parser.add_argument("--d_ff", type=int, default=90)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--num_kernels", type=int, default=6)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gpu_type", type=str, default="cuda")
    parser.add_argument("--label_len", type=int, default=0)
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--freq", type=str, default="d")
    args = parser.parse_args()

    # 動態設置 enc_in, dec_in, c_out
    dataset = StockDataset(data_path=args.data_path, seq_len=args.seq_len, pred_len=args.pred_len, split="train")
    args.enc_in = dataset.get_enc_in()
    args.dec_in = args.enc_in
    args.c_out = args.enc_in

    args.data_loader = StockDataset
    exp = Exp_Stock_Forecast(args)
    exp.train()

if __name__ == "__main__":
    main()