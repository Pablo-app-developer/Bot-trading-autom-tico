import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import ta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingModel:
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = StandardScaler()
        
    def prepare_data(self, df):
        """
        Prepara los datos para el modelo añadiendo indicadores técnicos
        """
        # Añadir indicadores técnicos
        df['RSI'] = ta.momentum.RSIIndicator(df['close']).rsi()
        df['MACD'] = ta.trend.MACD(df['close']).macd()
        df['BB_upper'] = ta.volatility.BollingerBands(df['close']).bollinger_hband()
        df['BB_lower'] = ta.volatility.BollingerBands(df['close']).bollinger_lband()
        
        # Calcular retornos
        df['returns'] = df['close'].pct_change()
        
        # Eliminar filas con valores NaN
        df = df.dropna()
        
        # Normalizar datos
        features = ['open', 'high', 'low', 'close', 'volume', 'RSI', 'MACD', 'BB_upper', 'BB_lower', 'returns']
        scaled_data = self.scaler.fit_transform(df[features])
        
        return scaled_data, features
    
    def create_sequences(self, data):
        """
        Crea secuencias para el modelo LSTM
        """
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            # Predecir si el precio subirá (1) o bajará (0)
            y.append(1 if data[i + self.sequence_length][3] > data[i + self.sequence_length - 1][3] else 0)
        return np.array(X), np.array(y)
    
    def build_model(self):
        """
        Construye el modelo LSTM
        """
        self.model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(self.sequence_length, 10)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("Modelo construido exitosamente")
    
    def train(self, df, epochs=50, batch_size=32, validation_split=0.2):
        """
        Entrena el modelo con los datos proporcionados
        """
        if self.model is None:
            self.build_model()
            
        scaled_data, _ = self.prepare_data(df)
        X, y = self.create_sequences(scaled_data)
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        logger.info("Entrenamiento completado")
        return history
    
    def predict(self, df):
        """
        Realiza predicciones sobre nuevos datos
        """
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado")
            
        scaled_data, _ = self.prepare_data(df)
        X, _ = self.create_sequences(scaled_data)
        
        predictions = self.model.predict(X)
        return predictions
    
    def evaluate_trading_decision(self, prediction, threshold=0.6):
        """
        Evalúa la decisión de trading basada en la predicción
        """
        if prediction > threshold:
            return "COMPRA"
        elif prediction < (1 - threshold):
            return "VENTA"
        else:
            return "MANTENER" 