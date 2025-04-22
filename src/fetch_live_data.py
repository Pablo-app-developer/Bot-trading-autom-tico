import requests
import pandas as pd
import os
import json
import datetime
import time
from dotenv import load_dotenv
import numpy as np
import talib

# Cargar variables de entorno
load_dotenv()

class OandaAPI:
    def __init__(self):
        self.api_key = os.getenv("OANDA_API_KEY")
        self.account_id = os.getenv("OANDA_ACCOUNT_ID")
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        self.base_url = "https://api-fxpractice.oanda.com/v3"  # URL para práctica (sandbox)
        # self.base_url = "https://api-fxtrade.oanda.com/v3"  # URL para cuentas reales
    
    def check_connection(self):
        """Verifica la conexión con la API de OANDA"""
        if not self.api_key or not self.account_id:
            print("Error: OANDA_API_KEY y OANDA_ACCOUNT_ID deben estar configurados en el archivo .env")
            return False
            
        url = f"{self.base_url}/accounts/{self.account_id}/summary"
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                print("Conexión con OANDA establecida correctamente.")
                return True
            else:
                print(f"Error al conectar con OANDA: {response.status_code}")
                print(response.text)
                return False
        except Exception as e:
            print(f"Error de conexión: {e}")
            return False
    
    def get_candles(self, instrument="EUR_USD", count=100, granularity="M5"):
        """
        Obtiene velas (OHLC) históricas para un instrumento
        
        Args:
            instrument: Par de divisas (por defecto EUR_USD)
            count: Número de velas a obtener
            granularity: Intervalo de tiempo (S5, M1, M5, M15, M30, H1, H4, D, W, M)
            
        Returns:
            DataFrame con los datos OHLC
        """
        url = f"{self.base_url}/instruments/{instrument}/candles"
        params = {
            'count': count,
            'granularity': granularity,
            'price': 'M'  # M = midpoint (promedio de bid y ask)
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code != 200:
                print(f"Error al obtener datos: {response.status_code}")
                print(response.text)
                return None
                
            data = response.json()
            
            # Procesar datos de velas
            candles = []
            for candle in data['candles']:
                if candle['complete']:  # Solo considerar velas completas
                    candle_data = {
                        'datetime': candle['time'],
                        'open': float(candle['mid']['o']),
                        'high': float(candle['mid']['h']),
                        'low': float(candle['mid']['l']),
                        'close': float(candle['mid']['c']),
                        'volume': int(candle['volume'])
                    }
                    candles.append(candle_data)
            
            # Convertir a DataFrame
            df = pd.DataFrame(candles)
            if not df.empty:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Error al obtener velas: {e}")
            return None
    
    def calculate_indicators(self, df):
        """
        Calcula indicadores técnicos para el DataFrame de velas
        
        Args:
            df: DataFrame con datos OHLC
            
        Returns:
            DataFrame con indicadores añadidos
        """
        if df is None or df.empty:
            return None
            
        # Hacer una copia para no modificar el original
        df_with_indicators = df.copy()
        
        # Calcular RSI (Relative Strength Index)
        df_with_indicators['rsi_14'] = talib.RSI(df_with_indicators['close'], timeperiod=14)
        
        # Calcular MACD (Moving Average Convergence Divergence)
        macd, macd_signal, _ = talib.MACD(df_with_indicators['close'], 
                                         fastperiod=12, 
                                         slowperiod=26, 
                                         signalperiod=9)
        df_with_indicators['macd'] = macd
        df_with_indicators['macd_signal'] = macd_signal
        
        # Calcular Bandas de Bollinger
        upper, middle, lower = talib.BBANDS(df_with_indicators['close'], 
                                           timeperiod=20, 
                                           nbdevup=2, 
                                           nbdevdn=2)
        df_with_indicators['bollinger_upper'] = upper
        df_with_indicators['bollinger_mid'] = middle
        df_with_indicators['bollinger_lower'] = lower
        
        # ATR (Average True Range)
        df_with_indicators['atr_14'] = talib.ATR(df_with_indicators['high'], 
                                                df_with_indicators['low'], 
                                                df_with_indicators['close'], 
                                                timeperiod=14)
        
        # CCI (Commodity Channel Index)
        df_with_indicators['cci_14'] = talib.CCI(df_with_indicators['high'], 
                                               df_with_indicators['low'], 
                                               df_with_indicators['close'], 
                                               timeperiod=14)
        
        # ADX (Average Directional Index)
        df_with_indicators['adx_14'] = talib.ADX(df_with_indicators['high'], 
                                               df_with_indicators['low'], 
                                               df_with_indicators['close'], 
                                               timeperiod=14)
        
        # Stochastic
        stoch_k, stoch_d = talib.STOCH(df_with_indicators['high'], 
                                     df_with_indicators['low'], 
                                     df_with_indicators['close'], 
                                     fastk_period=14, 
                                     slowk_period=3, 
                                     slowk_matype=0, 
                                     slowd_period=3, 
                                     slowd_matype=0)
        df_with_indicators['stoch_k'] = stoch_k
        df_with_indicators['stoch_d'] = stoch_d
        
        # Eliminar filas con valores NaN (por los indicadores que requieren datos históricos)
        df_with_indicators = df_with_indicators.dropna()
        
        return df_with_indicators

    def get_latest_data_with_indicators(self, instrument="EUR_USD", count=100, granularity="M5"):
        """
        Obtiene los datos más recientes con indicadores técnicos
        
        Returns:
            DataFrame con datos e indicadores técnicos
        """
        # Obtener velas
        candles_df = self.get_candles(instrument, count, granularity)
        
        if candles_df is None or candles_df.empty:
            return None
            
        # Calcular indicadores
        data_with_indicators = self.calculate_indicators(candles_df)
        
        return data_with_indicators

def guardar_datos_recientes(oanda_api, ruta_salida, instrument="EUR_USD", count=200, granularity="M5"):
    """
    Obtiene datos recientes y los guarda en un archivo CSV
    """
    data = oanda_api.get_latest_data_with_indicators(instrument, count, granularity)
    
    if data is not None and not data.empty:
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
        
        # Guardar datos
        data.to_csv(ruta_salida)
        print(f"Datos guardados en {ruta_salida}")
        return True
    
    print("No se pudieron obtener o procesar datos")
    return False

def obtener_ultima_fila(oanda_api, instrument="EUR_USD", count=100, granularity="M5"):
    """
    Obtiene solo la última fila de datos con indicadores para predicción
    """
    data = oanda_api.get_latest_data_with_indicators(instrument, count, granularity)
    
    if data is not None and not data.empty:
        # Tomar solo la última fila (datos más recientes)
        ultima_fila = data.iloc[-1:].copy()
        ultima_fila.reset_index(inplace=True)  # Convertir el índice en columna
        return ultima_fila
    
    return None

if __name__ == "__main__":
    # Inicializar API
    oanda = OandaAPI()
    
    # Verificar conexión
    if not oanda.check_connection():
        print("No se pudo establecer conexión con OANDA. Verifique sus credenciales.")
        exit(1)
    
    # Obtener y guardar datos recientes
    guardar_datos_recientes(oanda, "data/market/eurusd_live.csv")
    
    # Ejemplo de obtención de la última fila para predicción
    ultima_fila = obtener_ultima_fila(oanda)
    if ultima_fila is not None:
        print("\nDatos más recientes para predicción:")
        print(ultima_fila)
    else:
        print("No se pudieron obtener datos para predicción") 