import pandas as pd
import os

def cargar_datos_txt(ruta_archivo_txt, ruta_archivo_csv):
    print(f"📂 Leyendo archivo: {ruta_archivo_txt}")

    columnas = ["ticker", "date", "time", "open", "high", "low", "close", "volume"]
    try:
        df = pd.read_csv(ruta_archivo_txt, names=columnas, dtype=str)

        # Combinar y convertir fecha y hora
        df["datetime"] = pd.to_datetime(df["date"] + df["time"], format="%Y%m%d%H%M%S", errors="coerce")

        # Eliminar filas con errores de datetime
        df = df.dropna(subset=["datetime"])

        # Convertir columnas numéricas con control de errores
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Eliminar filas con valores numéricos inválidos
        df = df.dropna(subset=["open", "high", "low", "close", "volume"])

        # Reordenar columnas
        df = df[["datetime", "open", "high", "low", "close", "volume"]]

        # Guardar
        os.makedirs(os.path.dirname(ruta_archivo_csv), exist_ok=True)
        df.to_csv(ruta_archivo_csv, index=False)

        print(f"Archivo convertido y guardado: {ruta_archivo_csv}")
    except Exception as e:
        print(f"Error al procesar el archivo: {e}")

# Punto de entrada
if __name__ == "__main__":
    ruta_txt = "data/raw/EURUSD2001-2025.txt"
    ruta_csv = "data/market/eurusd_historico_1min.csv"
    cargar_datos_txt(ruta_txt, ruta_csv)
