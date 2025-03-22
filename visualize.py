import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def visualize_results():
    # 載入數據
    preds = np.load('results/preds.npy')
    trues = np.load('results/trues.npy')
    losses = pd.read_csv('results/losses.csv')

    # 載入原始數據以獲取日期
    df_raw = pd.read_csv('data/raw/aapl_daily.csv')
    dates = pd.to_datetime(df_raw['date'])

    # 假設測試數據從最後30天開始
    test_dataset = Dataset_Custom('data/raw', flag='test', size=[30, 5], data_path='aapl_daily.csv')
    test_dates = test_dataset.get_dates(-1)  # 獲取最後一個序列的日期

    # 創建子圖：1個股價圖 + 1個損失曲線圖
    fig = make_subplots(rows=2, cols=1, subplot_titles=("AAPL Stock Price Prediction", "Training Loss Curve"))

    # 繪製股價圖
    # 歷史股價（過去30天）
    fig.add_trace(
        go.Scatter(
            x=test_dates[:30],
            y=df_raw['close'].values[-35:-5],
            mode='lines',
            name='Historical Price'
        ),
        row=1, col=1
    )

    # 真實股價（未來5天）
    fig.add_trace(
        go.Scatter(
            x=test_dates[30:],
            y=trues[-1],
            mode='lines',
            name='True Price'
        ),
        row=1, col=1
    )

    # 預測股價（未來5天）
    fig.add_trace(
        go.Scatter(
            x=test_dates[30:],
            y=preds[-1],
            mode='lines',
            name='Predicted Price'
        ),
        row=1, col=1
    )

    # 繪製損失曲線
    fig.add_trace(
        go.Scatter(
            x=losses['epoch'],
            y=losses['train_loss'],
            mode='lines',
            name='Train Loss'
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=losses['epoch'],
            y=losses['val_loss'],
            mode='lines',
            name='Val Loss'
        ),
        row=2, col=1
    )

    # 設置圖表佈局
    fig.update_layout(
        height=800,
        template='plotly_dark',
        showlegend=True
    )

    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_yaxes(title_text="Loss", row=2, col=1)

    # 顯示圖表
    fig.show()


if __name__ == "__main__":
    visualize_results()