from data_provider.data_loader import StockDataset
from torch.utils.data import DataLoader

data_dict = {
    'stock': StockDataset,
}

def data_provider(args, flag):
    if args.data not in data_dict:
        raise ValueError(f"Dataset {args.data} not supported. Available datasets: {list(data_dict.keys())}")

    Data = data_dict[args.data]
    shuffle_flag = False if (flag == 'test' or flag == 'TEST' or flag == 'predict') else True
    drop_last = False  # 確保不丟棄最後一個批次
    batch_size = args.batch_size if flag != 'predict' else 1  # predict 模式使用 batch_size=1

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target='Close',  # 固定為 Close，與 StockDataset 一致
        timeenc=args.timeenc,
        freq=args.freq
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )
    return data_set, data_loader