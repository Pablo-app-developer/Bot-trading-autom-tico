import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import ta
import os
import datetime
import time
from typing import Optional

class MetaTraderAPI:
    def __init__(self):
        """Inicializa la conexión con MetaTrader 5"""
        self.connected = False
        self.initialize()
    
    def initialize(self):
        """Inicializa la conexión con MetaTrader 5"""
        if not mt5.initialize():
            print(f"Error al inicializar MetaTrader 5: {mt5.last_error()}")
            return False
        
        print(f"MetaTrader 5 conectado. Versión: {mt5.version()}")
        self.connected = True
        return True
    
    def check_connection(self):
        """Verifica si MetaTrader está conectado"""
        if not self.connected:
            return self.initialize()
        return True
    
    def shutdown(self):
        """Cierra la conexión con MetaTrader 5"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            print("Conexión con MetaTrader 5 cerrada.")
    
    def get_candles(self, symbol="EURUSD", timeframe=mt5.TIMEFRAME_M5, count=100):
        """
        Obtiene velas (OHLC) históricas para un símbolo
        
        Args:
            symbol: Par de divisas (por defecto EURUSD)
            timeframe: Intervalo de tiempo (mt5.TIMEFRAME_M1, mt5.TIMEFRAME_M5, etc.)
            count: Número de velas a obtener
            
        Returns:
            DataFrame con los datos OHLC
        """
        if not self.check_connection():
            return None
        
        # Verificar que el símbolo existe
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"Error: No se pudo encontrar el símbolo {symbol}")
            return None
        
        # Asegurarse que el símbolo está disponible para trading
        if not symbol_info.visible:
            print(f"El símbolo {symbol} no está visible, intentando habilitarlo...")
            if not mt5.symbol_select(symbol, True):
                print(f"Error: No se pudo habilitar el símbolo {symbol}")
                return None
        
        # Mapeo de timeframes de string a constantes de MT5
        timeframe_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }
        
        # Si timeframe es string, convertirlo a constante MT5
        if isinstance(timeframe, str):
            if timeframe in timeframe_map:
                timeframe = timeframe_map[timeframe]
            else:
                print(f"Error: Timeframe {timeframe} no soportado")
                return None
        
        # Obtener datos históricos
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        
        if rates is None or len(rates) == 0:
            print(f"Error al obtener datos históricos: {mt5.last_error()}")
            return None
        
        # Convertir a DataFrame
        df = pd.DataFrame(rates)
        
        # Convertir timestamp a datetime
        df['datetime'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('datetime', inplace=True)
        
        # Renombrar columnas para consistencia con el código existente
        df.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'tick_volume': 'volume'
        }, inplace=True)
        
        return df
    
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
        df_with_indicators['rsi_14'] = ta.momentum.RSIIndicator(df_with_indicators['close'], window=14).rsi()
        
        # Calcular MACD (Moving Average Convergence Divergence)
        macd_indicator = ta.trend.MACD(df_with_indicators['close'], window_slow=26, window_fast=12, window_sign=9)
        df_with_indicators['macd'] = macd_indicator.macd()
        df_with_indicators['macd_signal'] = macd_indicator.macd_signal()
        
        # Calcular Bandas de Bollinger
        bollinger = ta.volatility.BollingerBands(df_with_indicators['close'], window=20, window_dev=2)
        df_with_indicators['bollinger_upper'] = bollinger.bollinger_hband()
        df_with_indicators['bollinger_mid'] = bollinger.bollinger_mavg()
        df_with_indicators['bollinger_lower'] = bollinger.bollinger_lband()
        
        # ATR (Average True Range)
        df_with_indicators['atr_14'] = ta.volatility.AverageTrueRange(df_with_indicators['high'], 
                                                df_with_indicators['low'], 
                                                df_with_indicators['close'], 
                                                window=14).average_true_range()
        
        # CCI (Commodity Channel Index)
        df_with_indicators['cci_14'] = ta.trend.CCIIndicator(df_with_indicators['high'], 
                                               df_with_indicators['low'], 
                                               df_with_indicators['close'], 
                                               window=14).cci()
        
        # ADX (Average Directional Index)
        df_with_indicators['adx_14'] = ta.trend.ADXIndicator(df_with_indicators['high'], 
                                               df_with_indicators['low'], 
                                               df_with_indicators['close'], 
                                               window=14).adx()
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df_with_indicators['high'], 
                                     df_with_indicators['low'], 
                                     df_with_indicators['close'], 
                                     window=14, smooth_window=3)
        df_with_indicators['stoch_k'] = stoch.stoch()
        df_with_indicators['stoch_d'] = stoch.stoch_signal()
        
        # Eliminar filas con valores NaN (por los indicadores que requieren datos históricos)
        df_with_indicators = df_with_indicators.dropna()
        
        return df_with_indicators
    
    def get_latest_data_with_indicators(self, symbol="EURUSD", timeframe=mt5.TIMEFRAME_M5, count=100):
        """
        Obtiene los datos más recientes con indicadores técnicos
        
        Returns:
            DataFrame con datos e indicadores técnicos
        """
        # Obtener velas
        candles_df = self.get_candles(symbol, timeframe, count)
        
        if candles_df is None or candles_df.empty:
            return None
            
        # Calcular indicadores
        data_with_indicators = self.calculate_indicators(candles_df)
        
        return data_with_indicators

def guardar_datos_recientes(mt_api, ruta_salida, symbol="EURUSD", timeframe=mt5.TIMEFRAME_M5, count=200):
    """
    Obtiene datos recientes y los guarda en un archivo CSV
    """
    data = mt_api.get_latest_data_with_indicators(symbol, timeframe, count)
    
    if data is not None and not data.empty:
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
        
        # Guardar datos
        data.to_csv(ruta_salida)
        print(f"Datos guardados en {ruta_salida}")
        return True
    
    print("No se pudieron obtener o procesar datos")
    return False

def obtener_ultima_fila(mt_api, symbol="EURUSD", timeframe=mt5.TIMEFRAME_M5, count=100):
    """
    Obtiene solo la última fila de datos con indicadores para predicción
    """
    data = mt_api.get_latest_data_with_indicators(symbol, timeframe, count)
    
    if data is not None and not data.empty:
        # Tomar solo la última fila (datos más recientes)
        ultima_fila = data.iloc[-1:].copy()
        ultima_fila.reset_index(inplace=True)  # Convertir el índice en columna
        return ultima_fila
    
    return None

if __name__ == "__main__":
    # Inicializar API
    mt = MetaTraderAPI()
    
    # Verificar conexión
    if not mt.check_connection():
        print("No se pudo establecer conexión con MetaTrader 5. Verifique su instalación.")
        exit(1)
    
    # Obtener y guardar datos recientes (usando M5 - 5 minutos)
    guardar_datos_recientes(mt, "data/market/eurusd_live.csv", timeframe="M5")
    
    # Ejemplo de obtención de la última fila para predicción
    ultima_fila = obtener_ultima_fila(mt)
    if ultima_fila is not None:
        print("\nDatos más recientes para predicción:")
        print(ultima_fila)
    else:
        print("No se pudieron obtener datos para predicción")
    
    # Cerrar conexión
    mt.shutdown() 