import yfinance as yf
import pandas as pd
import os

def fetch_stock_data(ticker="AAPL", period="1y", output_dir="data/raw"):
    os.makedirs(output_dir, exist_ok=True)
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df = df[["Open", "High", "Low", "Close", "Volume"]].reset_index()
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    output_path = os.path.join(output_dir, f"{ticker.lower()}_daily.csv")
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    fetch_stock_data()