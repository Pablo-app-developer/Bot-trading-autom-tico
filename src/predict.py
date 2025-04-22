import pandas as pd
import numpy as np
import joblib
import os
import datetime
import time
import sys
# Agregar el directorio src al path para importar correctamente
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fetch_metatrader_data import MetaTraderAPI, obtener_ultima_fila

def cargar_modelo(ruta_modelo):
    """
    Carga el modelo entrenado y el codificador de etiquetas
    """
    print(f"Cargando modelo desde {ruta_modelo}...")
    modelo_data = joblib.load(ruta_modelo)
    model = modelo_data['model']
    label_encoder = modelo_data['label_encoder']
    return model, label_encoder

def preprocesar_datos_tiempo_real(datos, columnas_modelo):
    """
    Preprocesa los datos de tiempo real para que coincidan con el formato 
    esperado por el modelo
    """
    # Asegurarse que los datos tienen las mismas columnas que el modelo espera
    # Seleccionar solo las columnas que espera el modelo
    columnas_presentes = [col for col in columnas_modelo if col in datos.columns]
    datos_proc = datos[columnas_presentes].copy()
    
    # Agregar columnas faltantes con valores 0
    for col in columnas_modelo:
        if col not in datos_proc.columns:
            print(f"Advertencia: Columna {col} no está presente en los datos. Se añadirá con valores 0.")
            datos_proc[col] = 0
    
    # Reordenar columnas para que coincidan con el orden esperado por el modelo
    datos_proc = datos_proc[columnas_modelo]
    
    # Rellenar valores faltantes con la media
    for col in datos_proc.columns:
        if datos_proc[col].isnull().any():
            datos_proc[col].fillna(datos_proc[col].mean(), inplace=True)
    
    return datos_proc

def obtener_datos_tiempo_real(simular=False):
    """
    Obtiene los datos más recientes del EURUSD desde MetaTrader o utiliza simulación
    """
    if simular:
        print("SIMULACIÓN: Obteniendo datos simulados...")
        
        # Simulación de datos para demostración
        datos_simulados = {
            'open': [1.0756],
            'high': [1.0782],
            'low': [1.0748],
            'close': [1.0769],
            'rsi_14': [56.78],
            'macd': [0.00023],
            'macd_signal': [0.00018],
            'bollinger_upper': [1.0805],
            'bollinger_mid': [1.0765],
            'bollinger_lower': [1.0725],
            'atr_14': [0.0043],
            'cci_14': [105.32],
            'adx_14': [23.45],
            'stoch_k': [72.34],
            'stoch_d': [68.21],
        }
        
        # Crear DataFrame con timestamp actual
        df = pd.DataFrame(datos_simulados)
        df['datetime'] = datetime.datetime.now()
        
        return df
    else:
        print("Obteniendo datos reales de MetaTrader...")
        try:
            # Inicializar API de MetaTrader
            mt_api = MetaTraderAPI()
            
            # Verificar conexión
            if not mt_api.check_connection():
                print("No se pudo establecer conexión con MetaTrader. Usando datos simulados.")
                return obtener_datos_tiempo_real(simular=True)
            
            # Obtener datos reales
            ultima_fila = obtener_ultima_fila(mt_api)
            
            # Cerrar conexión
            mt_api.shutdown()
            
            if ultima_fila is None:
                print("No se pudieron obtener datos reales. Usando datos simulados.")
                return obtener_datos_tiempo_real(simular=True)
                
            return ultima_fila
        except Exception as e:
            print(f"Error al obtener datos reales: {e}")
            print("Usando datos simulados como respaldo.")
            return obtener_datos_tiempo_real(simular=True)

def realizar_prediccion(model, label_encoder, datos_preprocesados):
    """
    Realiza la predicción usando el modelo cargado
    """
    prediccion_encoded = model.predict(datos_preprocesados)
    prediccion_clase = label_encoder.inverse_transform(prediccion_encoded)
    
    probabilidades = model.predict_proba(datos_preprocesados)
    confianza = np.max(probabilidades, axis=1)[0]
    
    return prediccion_clase[0], confianza

def interpretar_prediccion(prediccion_clase, confianza):
    """
    Interpreta la predicción y devuelve una recomendación
    """
    direccion = {
        -1: "BAJADA (VENDER)",
        0: "NEUTRAL (MANTENER)",
        1: "SUBIDA (COMPRAR)"
    }
    
    mensaje = f"Predicción: {direccion.get(prediccion_clase, 'DESCONOCIDO')}"
    mensaje += f" | Confianza: {confianza*100:.2f}%"
    
    return mensaje

def ejecutar_prediccion_continua(ruta_modelo, intervalo_segundos=60, columnas_modelo=None, simular=False):
    """
    Realiza predicciones continuas en intervalos regulares
    """
    # Cargar modelo y encoder
    model, label_encoder = cargar_modelo(ruta_modelo)
    
    # Si no se proporcionan las columnas del modelo, usar nombres genéricos
    if columnas_modelo is None:
        # Obtener las columnas del modelo - esto debe coincidir con las columnas usadas en entrenamiento
        x_train_path = os.path.join(os.path.dirname(ruta_modelo), 'X_train.csv')
        if os.path.exists(x_train_path):
            columnas_modelo = pd.read_csv(x_train_path, nrows=0).columns.tolist()
        else:
            raise ValueError("No se encontraron las columnas del modelo. Especifica columnas_modelo.")
    
    print(f"Iniciando predicciones cada {intervalo_segundos} segundos. Presiona Ctrl+C para detener.")
    
    try:
        while True:
            # Obtener datos recientes
            datos_recientes = obtener_datos_tiempo_real(simular=simular)
            
            # Preprocesar datos
            datos_preprocesados = preprocesar_datos_tiempo_real(datos_recientes, columnas_modelo)
            
            # Realizar predicción
            prediccion_clase, confianza = realizar_prediccion(model, label_encoder, datos_preprocesados)
            
            # Interpretar y mostrar resultado
            resultado = interpretar_prediccion(prediccion_clase, confianza)
            print(f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {resultado}")
            
            # Guardar en archivo histórico
            try:
                with open("data/market/predicciones_historico.csv", "a") as f:
                    if os.path.getsize("data/market/predicciones_historico.csv") == 0:
                        f.write("timestamp,precio,prediccion,confianza\n")
                    f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')},"
                            f"{datos_recientes['close'].values[0]},"
                            f"{prediccion_clase},"
                            f"{confianza}\n")
            except Exception as e:
                print(f"Error al guardar predicción en histórico: {e}")
            
            # Esperar para la siguiente predicción
            time.sleep(intervalo_segundos)
            
    except KeyboardInterrupt:
        print("Deteniendo predicciones...")
    except Exception as e:
        print(f"Error durante la predicción: {e}")

if __name__ == "__main__":
    # Ubicación del modelo entrenado
    ruta_modelo = "data/model/modelo_clasificacion.pkl"
    
    # Asegurarse que el directorio para históricos existe
    os.makedirs("data/market", exist_ok=True)
    
    # Ejemplo de uso para una predicción única
    model, label_encoder = cargar_modelo(ruta_modelo)
    
    # Determinar si usar datos simulados o reales
    usar_simulacion = False  # Usar datos reales de MetaTrader
    datos_recientes = obtener_datos_tiempo_real(simular=usar_simulacion)
    
    # Cargar columnas del modelo desde X_train para asegurar consistencia
    x_train_path = os.path.join(os.path.dirname(ruta_modelo), 'X_train.csv')
    columnas_modelo = pd.read_csv(x_train_path, nrows=0).columns.tolist()
    
    # Preprocesar
    datos_preprocesados = preprocesar_datos_tiempo_real(datos_recientes, columnas_modelo)
    
    # Predecir
    prediccion_clase, confianza = realizar_prediccion(model, label_encoder, datos_preprocesados)
    
    # Mostrar resultado
    resultado = interpretar_prediccion(prediccion_clase, confianza)
    print(f"\n[PREDICCIÓN ÚNICA] {resultado}")
    
    # Para predicciones continuas, descomenta la siguiente línea:
    # ejecutar_prediccion_continua(ruta_modelo, intervalo_segundos=60, simular=usar_simulacion) 