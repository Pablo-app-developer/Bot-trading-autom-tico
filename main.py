from src.fetch_data import get_eurusd_data

if __name__ == "__main__":
    df = get_eurusd_data()
    print(df.head())
