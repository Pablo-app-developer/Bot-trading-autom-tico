import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

def entrenar_modelo(ruta_entrada_dir, ruta_salida_dir, random_state=42):
    """
    Entrena un modelo de clasificación XGBoost y guarda el modelo
    """
    print("Entrenando modelo XGBoost...")
    
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
    
    # Transformar etiquetas para XGBoost: [-1, 0, 1] -> [0, 1, 2]
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Guardar el codificador de etiquetas para su uso posterior
    label_map = {original: transformed for original, transformed in zip(label_encoder.classes_, range(len(label_encoder.classes_)))}
    print(f"Mapa de transformación de etiquetas: {label_map}")
    
    # Crear directorio de salida si no existe
    os.makedirs(ruta_salida_dir, exist_ok=True)
    
    # Entrenar modelo XGBoost
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    
    model.fit(X_train, y_train_encoded)
    
    # Evaluar modelo
    y_pred_encoded = model.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)  # Convertir de vuelta a etiquetas originales
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Precisión del modelo: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nInforme de clasificación:")
    print(classification_report(y_test, y_pred))
    
    # Guardar modelo
    modelo_path = os.path.join(ruta_salida_dir, 'modelo_clasificacion.pkl')
    joblib.dump({
        'model': model,
        'label_encoder': label_encoder
    }, modelo_path)
    
    # Guardar métricas
    with open(os.path.join(ruta_salida_dir, 'metricas.txt'), 'w') as f:
        f.write(f"Precisión: {accuracy:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n\n")
        f.write("Importancia de características:\n")
        importance_scores = model.feature_importances_
        feature_importance = sorted(zip(X_train.columns, importance_scores), 
                               key=lambda x: x[1], reverse=True)
        for feature, importance in feature_importance:
            f.write(f"{feature}: {importance:.4f}\n")
    
    print(f"Entrenamiento finalizado. Modelo guardado en {modelo_path}")
    return model

# Ejecución directa
if __name__ == "__main__":
    entrenar_modelo(
        ruta_entrada_dir="data/model",
        ruta_salida_dir="data/model"
    )
