import yfinance as yf
import pandas as pd
import os


def fetch_stock_data(ticker="AAPL", start_date="2024-03-22", end_date="2025-03-22"):
    # 創建儲存目錄
    os.makedirs("data/raw", exist_ok=True)

    # 抓取數據
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date, interval="1d")

    # 僅保留日期和收盤價
    data = data[['Close']].reset_index()
    data.columns = ['date', 'close']

    # 儲存數據
    data.to_csv(f"data/raw/{ticker.lower()}_daily.csv", index=False)
    print(f"Data for {ticker} saved to data/raw/{ticker.lower()}_daily.csv")


if __name__ == "__main__":
    fetch_stock_data()