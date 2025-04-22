import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import os

def fetch_market_data():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)

    df = yf.download("EURUSD=X", start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval="1d")
    df.reset_index(inplace=True)

    os.makedirs("data/market", exist_ok=True)
    file_path = f"data/market/eurusd_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(file_path, index=False)
    print(f"Datos de mercado guardados en: {file_path}")

if __name__ == "__main__":
    fetch_market_data()
