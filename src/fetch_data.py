import yfinance as yf
import pandas as pd
import os

def get_eurusd_data(start="2010-01-01", end="2024-12-31"):
    df = yf.download("EURUSD=X", start=start, end=end, interval="1d")
    df.to_csv("eurusd_bot/data/eurusd.csv")
    return df

