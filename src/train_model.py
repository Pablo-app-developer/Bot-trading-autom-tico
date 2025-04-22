import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

def entrenar_modelo(ruta_entrada_dir, ruta_salida_dir, random_state=42):
    """
    Entrena un modelo de clasificación Random Forest y guarda el modelo
    """
    print("Entrenando modelo de clasificación...")
    
    # Verificar que los archivos existen
    x_train_path = os.path.join(ruta_entrada_dir, 'X_train.csv')
    y_train_path = os.path.join(ruta_entrada_dir, 'y_train.csv')
    x_test_path = os.path.join(ruta_entrada_dir, 'X_test.csv')
    y_test_path = os.path.join(ruta_entrada_dir, 'y_test.csv')
    
    if not all(os.path.exists(p) for p in [x_train_path, y_train_path, x_test_path, y_test_path]):
        raise FileNotFoundError(f"No se pudieron encontrar los archivos de datos en {ruta_entrada_dir}")
    
    # Cargar datos
    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()
    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()
    
    # Crear directorio de salida si no existe
    os.makedirs(ruta_salida_dir, exist_ok=True)
    
    # Entrenar modelo
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)
    
    # Evaluar modelo
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Precisión del modelo: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nInforme de clasificación:")
    print(classification_report(y_test, y_pred))
    
    # Guardar modelo
    modelo_path = os.path.join(ruta_salida_dir, 'modelo_clasificacion.pkl')
    joblib.dump(model, modelo_path)
    
    # Guardar métricas
    metricas = {
        'accuracy': accuracy,
        'f1_score': f1,
        'feature_importance': dict(zip(X_train.columns, model.feature_importances_))
    }
    
    # Guardar en formato legible
    with open(os.path.join(ruta_salida_dir, 'metricas.txt'), 'w') as f:
        f.write(f"Precisión: {accuracy:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n\n")
        f.write("Importancia de características:\n")
        for feature, importance in sorted(zip(X_train.columns, model.feature_importances_), 
                                         key=lambda x: x[1], reverse=True):
            f.write(f"{feature}: {importance:.4f}\n")
    
    print(f"Entrenamiento finalizado. Modelo guardado en {modelo_path}")
    return model

# Ejecución directa
if __name__ == "__main__":
    entrenar_modelo(
        ruta_entrada_dir="data/model",
        ruta_salida_dir="data/model"
    )
