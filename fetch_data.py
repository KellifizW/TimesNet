import yfinance as yf
import pandas as pd
import os

def fetch_stock_data(ticker="AAPL", years=5, output_dir="data/raw"):
    try:
        # 創建輸出目錄
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{ticker.lower()}_daily.csv")

        # 下載股票數據
        print(f"正在下載 {ticker} 的股票數據（{years} 年）...")
        stock = yf.Ticker(ticker)
        df = stock.history(period=f"{years}y")

        # 檢查數據是否為空
        if df.empty:
            raise ValueError(f"無法下載 {ticker} 的數據：返回的數據為空。請檢查股票代號或網絡連接。")

        # 處理數據
        df = df[["Open", "High", "Low", "Close", "Volume"]].reset_index()
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
        df.to_csv(output_path, index=False)
        print(f"數據已保存到 {output_path}")

        # 檢查文件是否真的生成
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"數據文件 {output_path} 未生成，請檢查寫入權限或磁盤空間。")

        return output_path

    except Exception as e:
        print(f"下載 {ticker} 數據時發生錯誤：{str(e)}")
        raise  # 確保異常被拋出，不返回 None

if __name__ == "__main__":
    try:
        years = int(input("請輸入下載數據的年份（預設: 5）：") or 5)
        fetch_stock_data(years=years)
    except ValueError:
        print("請輸入有效的數字！")
        exit(1)
    except Exception as e:
        print(f"執行 fetch_stock_data 時發生錯誤：{str(e)}")
        exit(1)