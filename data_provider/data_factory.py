from data_provider.data_loader import StockDataset
from torch.utils.data import DataLoader

data_dict = {
    'stock': StockDataset,
}

def data_provider(args, flag):
    if args.data not in data_dict:
        raise ValueError(f"Dataset {args.data} not supported. Available datasets: {list(data_dict.keys())}")

    Data = data_dict[args.data]
    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size

    data_set = Data(
        data_path=args.data_path,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        split=flag
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader