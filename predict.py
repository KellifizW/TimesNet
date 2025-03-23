import argparse
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
    args = parser.parse_args()

    args.data_loader = StockDataset
    exp = Exp_Stock_Forecast(args)
    exp.predict()


if __name__ == "__main__":
    main()