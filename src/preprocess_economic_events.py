import pandas as pd
import os

def filtrar_eventos_economicos(archivo_entrada: str, archivo_salida: str) -> None:
    print(f"📂 Cargando archivo: {archivo_entrada}")

    if not os.path.exists(archivo_entrada):
        raise FileNotFoundError(f"El archivo '{archivo_entrada}' no existe.")

    df = pd.read_csv(archivo_entrada)

    # Eliminar espacios en blanco y valores vacíos en la columna de fecha
    df["Start"] = df["Start"].astype(str).str.strip()
    df = df[df["Start"].notnull() & (df["Start"] != "")]

    # Conversión estricta con formato fijo
    df["Start"] = pd.to_datetime(df["Start"], format="%m/%d/%Y %H:%M:%S", errors="coerce")

    # Eliminar fechas no convertidas
    df = df.dropna(subset=["Start"])

    # Filtrar por monedas relevantes
    df = df[df["Currency"].isin(["EUR", "USD"])]

    # Asignar sentimiento basado en el impacto
    df["sentiment"] = df["Impact"].map({
        "HIGH": "positive",
        "MEDIUM": "neutral"
    }).fillna("negative")

    os.makedirs(os.path.dirname(archivo_salida), exist_ok=True)
    df.to_csv(archivo_salida, index=False)
    print(f"Archivo procesado correctamente: {archivo_salida}")


if __name__ == "__main__":
    archivo_entrada = "data/raw/economic_events.csv"
    archivo_salida = "data/processed/economic_events_processed.csv"
    filtrar_eventos_economicos(archivo_entrada, archivo_salida)
