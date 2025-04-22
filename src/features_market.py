import os
import pandas as pd
import argparse

def calcular_retornos_y_clases(ruta_entrada, ruta_salida, umbral=0.001):
    if not os.path.exists(ruta_entrada):
        raise FileNotFoundError(f"❌ Archivo no encontrado: {ruta_entrada}")

    # Cargar datos
    df = pd.read_csv(ruta_entrada, parse_dates=["datetime"])
    df = df[["datetime", "open", "high", "low", "close"]].copy()
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Calcular retorno futuro a 1 hora (60 min)
    df["close_futuro"] = df["close"].shift(-60)
    df["future_return_1h"] = (df["close_futuro"] - df["close"]) / df["close"]

    # Asignar clase objetivo
    df["target_class_1h"] = df["future_return_1h"].apply(
        lambda x: 1 if x > umbral else (-1 if x < -umbral else 0)
    )

    df.dropna(inplace=True)
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    df.to_csv(ruta_salida, index=False)
    print(f"✅ Archivo procesado y guardado en: {ruta_salida}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generar targets de trading para 1H")
    parser.add_argument("--input", required=True, help="Ruta al CSV original (1min OHLC)")
    parser.add_argument("--output", default="data/processed/eurusd_with_targets.csv", help="Ruta de salida")
    args = parser.parse_args()

    calcular_retornos_y_clases(args.input, args.output)
