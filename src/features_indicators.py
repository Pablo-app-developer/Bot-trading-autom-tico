import pandas as pd
import ta

def agregar_indicadores_tecnicos(df: pd.DataFrame) -> pd.DataFrame:
    # RSI
    df["RSI"] = ta.momentum.RSIIndicator(close=df["close"], window=14).rsi()
    
    # MACD
    macd = ta.trend.MACD(close=df["close"])
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()

    # Bandas de Bollinger
    boll = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2)
    df["BB_upper"] = boll.bollinger_hband()
    df["BB_lower"] = boll.bollinger_lband()

    # SMA y EMA
    df["SMA_20"] = ta.trend.SMAIndicator(close=df["close"], window=20).sma_indicator()
    df["SMA_50"] = ta.trend.SMAIndicator(close=df["close"], window=50).sma_indicator()
    df["EMA_20"] = ta.trend.EMAIndicator(close=df["close"], window=20).ema_indicator()
    df["EMA_50"] = ta.trend.EMAIndicator(close=df["close"], window=50).ema_indicator()

    # ATR
    df["ATR"] = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).average_true_range()

    return df

if __name__ == "__main__":
    ruta_entrada = "data/processed/eurusd_with_targets.csv"
    ruta_salida = "data/processed/eurusd_with_indicators.csv"

    df = pd.read_csv(ruta_entrada, parse_dates=["datetime"])
    df = agregar_indicadores_tecnicos(df)

    df.dropna(inplace=True)  # eliminamos las primeras filas con NaN por los indicadores
    df.to_csv(ruta_salida, index=False)
    print(f"✅ Indicadores calculados y guardados en: {ruta_salida}")
