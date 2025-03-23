import argparse
from exp.exp_stock_forecast import Exp_Stock_Forecast
from data_provider.data_loader import StockDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="TimesNet")
    parser.add_argument("--data_path", type=str, default="data/raw/aapl_daily.csv")
    parser.add_argument("--seq_len", type=int, default=15)  # 論文中的 n_inputs
    parser.add_argument("--pred_len", type=int, default=5)
    parser.add_argument("--train_epochs", type=int, default=150)  # 論文中的 epochs
    parser.add_argument("--learning_rate", type=float, default=0.0001)  # 論文中的 learning_rate
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--d_model", type=int, default=90)  # 論文中的 hidden_size
    parser.add_argument("--d_ff", type=int, default=90)  # 論文中的 conv_hidden_size
    parser.add_argument("--top_k", type=int, default=5)  # 論文中的 top_k
    parser.add_argument("--num_kernels", type=int, default=6)  # 論文中的 num_kernels
    parser.add_argument("--e_layers", type=int, default=2)  # 論文中的 encoder_layers
    parser.add_argument("--dropout", type=float, default=0.1)  # 論文中的 dropout
    args = parser.parse_args()

    args.data_loader = StockDataset
    exp = Exp_Stock_Forecast(args)
    exp.train()


if __name__ == "__main__":
    main()