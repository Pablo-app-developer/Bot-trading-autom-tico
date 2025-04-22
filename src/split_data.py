import pandas as pd
from sklearn.model_selection import train_test_split
import os

def dividir_datos(ruta_entrada, ruta_salida_dir, test_size=0.2, random_state=42):
    """
    Divide el conjunto de datos procesado en entrenamiento y prueba
    """
    print("Dividiendo datos en conjuntos de entrenamiento y prueba...")
    
    # Verificar que el archivo de entrada existe
    if not os.path.exists(ruta_entrada):
        raise FileNotFoundError(f"No se pudo encontrar el archivo {ruta_entrada}")
        
    # Cargar datos procesados
    df = pd.read_csv(ruta_entrada)
    
    # Crear directorio de salida si no existe
    os.makedirs(ruta_salida_dir, exist_ok=True)
    
    # Dividir en características y variable objetivo
    X = df.drop(columns=['target_class_1h'], errors='ignore')
    y = df['target_class_1h']
    
    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Guardar conjuntos en archivos separados
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    train_df.to_csv(os.path.join(ruta_salida_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(ruta_salida_dir, 'test.csv'), index=False)
    
    print(f"División completada. Datos guardados en {ruta_salida_dir}")

# Ejecución directa
if __name__ == "__main__":
    dividir_datos(
        ruta_entrada="data/processed/eurusd_with_targets.csv",
        ruta_salida_dir="data/model"
    ) 