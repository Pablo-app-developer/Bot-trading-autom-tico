import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def preprocesar_datos(ruta_entrada, ruta_salida_dir, test_size=0.2, random_state=42):
    # Cargar datos procesados
    df = pd.read_csv(ruta_entrada, parse_dates=['datetime'])

    # Eliminar columnas innecesarias
    df = df.drop(columns=['datetime', 'volume'], errors='ignore')

    # Eliminar filas con valores faltantes (por los indicadores)
    df = df.dropna()

    # Separar variables independientes y target
    X = df.drop(columns=['target_class_1h'])
    y = df['target_class_1h']

    # Escalado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    # Separar en train y test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_df, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Crear carpeta si no existe
    os.makedirs(ruta_salida_dir, exist_ok=True)

    # Guardar datasets
    X_train.to_csv(os.path.join(ruta_salida_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(ruta_salida_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(ruta_salida_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(ruta_salida_dir, 'y_test.csv'), index=False)

    print(f"Preprocesamiento finalizado. Datos guardados en {ruta_salida_dir}")

# Ejecución directa
if __name__ == "__main__":
    preprocesar_datos(
        ruta_entrada="data/processed/eurusd_with_targets.csv",
        ruta_salida_dir="data/model"
    )
