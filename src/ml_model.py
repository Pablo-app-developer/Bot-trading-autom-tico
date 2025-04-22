import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Attention, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import ta
import logging
import matplotlib.pyplot as plt
from talib import CDLHAMMER, CDLENGULFING, CDLDOJI, CDLMORNINGSTAR, CDLEVENINGSTAR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingModel:
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        
    def prepare_data(self, df):
        """
        Prepara los datos para el modelo añadiendo indicadores técnicos avanzados
        """
        # Crear una copia para evitar warnings
        df = df.copy()
        
        # Indicadores básicos
        df['RSI'] = ta.momentum.RSIIndicator(df['close']).rsi()
        df['MACD'] = ta.trend.MACD(df['close']).macd()
        df['MACD_signal'] = ta.trend.MACD(df['close']).macd_signal()
        df['MACD_diff'] = ta.trend.MACD(df['close']).macd_diff()
        df['BB_upper'] = ta.volatility.BollingerBands(df['close']).bollinger_hband()
        df['BB_lower'] = ta.volatility.BollingerBands(df['close']).bollinger_lband()
        df['BB_width'] = df['BB_upper'] - df['BB_lower']
        
        # Indicadores avanzados de momentum
        df['ROC'] = ta.momentum.ROCIndicator(df['close']).roc()
        df['Stoch_k'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close']).stoch()
        df['Stoch_d'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close']).stoch_signal()
        df['Williams_R'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
        
        # Indicadores de volatilidad
        df['ATR'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        df['ATR_pct'] = df['ATR'] / df['close'] * 100  # ATR como porcentaje del precio
        
        # Indicadores de tendencia
        df['ADX'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
        df['CCI'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()
        df['Aroon_up'] = ta.trend.AroonIndicator(df['close']).aroon_up()
        df['Aroon_down'] = ta.trend.AroonIndicator(df['close']).aroon_down()
        df['Aroon_diff'] = df['Aroon_up'] - df['Aroon_down']
        
        # Patrones de velas (1 si se detecta el patrón, 0 si no)
        df_np = df.to_numpy()
        df['CDLHAMMER'] = CDLHAMMER(df_np, df_np, df_np, df_np)
        df['CDLENGULFING'] = CDLENGULFING(df_np, df_np, df_np, df_np)
        df['CDLDOJI'] = CDLDOJI(df_np, df_np, df_np, df_np)
        df['CDLMORNINGSTAR'] = CDLMORNINGSTAR(df_np, df_np, df_np, df_np)
        df['CDLEVENINGSTAR'] = CDLEVENINGSTAR(df_np, df_np, df_np, df_np)
        
        # Calcular retornos
        df['returns'] = df['close'].pct_change()
        df['returns_vol'] = df['returns'].rolling(window=20).std()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Características de volumen
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Creamos características derivadas
        df['close_ma50'] = df['close'].rolling(window=50).mean()
        df['close_ma200'] = df['close'].rolling(window=200).mean()
        df['ma_ratio'] = df['close_ma50'] / df['close_ma200']
        df['price_ma50_ratio'] = df['close'] / df['close_ma50']
        
        # Features adicionales
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Eliminar filas con valores NaN
        df = df.dropna()
        
        # Seleccionar características
        features = [
            'open', 'high', 'low', 'close', 'volume', 
            'RSI', 'MACD', 'MACD_signal', 'MACD_diff', 
            'BB_upper', 'BB_lower', 'BB_width',
            'ROC', 'Stoch_k', 'Stoch_d', 'Williams_R',
            'ATR', 'ATR_pct', 'ADX', 'CCI', 
            'Aroon_up', 'Aroon_down', 'Aroon_diff',
            'CDLHAMMER', 'CDLENGULFING', 'CDLDOJI', 'CDLMORNINGSTAR', 'CDLEVENINGSTAR',
            'returns', 'returns_vol', 'log_returns',
            'volume_ma', 'volume_ratio',
            'close_ma50', 'close_ma200', 'ma_ratio', 'price_ma50_ratio',
            'high_low_ratio', 'close_open_ratio'
        ]
        
        # Normalizar datos
        scaled_data = self.scaler.fit_transform(df[features])
        
        return scaled_data, features, df
    
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
    
    def build_attention_lstm_model(self, input_shape):
        """
        Construye un modelo LSTM apilado con mecanismo de atención
        """
        # Input layer
        inputs = Input(shape=input_shape)
        
        # Primera capa LSTM
        lstm1 = LSTM(128, return_sequences=True)(inputs)
        lstm1 = Dropout(0.2)(lstm1)
        
        # Segunda capa LSTM
        lstm2 = LSTM(64, return_sequences=True)(lstm1)
        lstm2 = Dropout(0.2)(lstm2)
        
        # Mecanismo de atención
        attention = Attention()([lstm2, lstm2])
        
        # Concatenar con salida LSTM
        concat = Concatenate()([lstm2, attention])
        
        # Tercera capa LSTM que procesa la salida con atención
        lstm3 = LSTM(32, return_sequences=False)(concat)
        lstm3 = Dropout(0.2)(lstm3)
        
        # Capas densas para la clasificación
        dense1 = Dense(16, activation='relu')(lstm3)
        output = Dense(1, activation='sigmoid')(dense1)
        
        # Crear y compilar modelo
        model = Model(inputs=inputs, outputs=output)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("Modelo con atención construido exitosamente")
        return model
    
    def build_model(self):
        """
        Construye un modelo LSTM avanzado
        """
        self.model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(self.sequence_length, self.num_features)),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("Modelo construido exitosamente")
    
    def time_series_cv_train(self, df, n_splits=5, epochs=50, batch_size=32):
        """
        Entrena el modelo utilizando validación cruzada para series temporales
        """
        scaled_data, features, _ = self.prepare_data(df)
        self.num_features = len(features)
        
        X, y = self.create_sequences(scaled_data)
        
        # Crear splits para validación cruzada de series temporales
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        fold_metrics = []
        all_histories = []
        
        # Configurar Early Stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info(f"Entrenando fold {fold+1}/{n_splits}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Para el primer fold, crear el modelo de atención
            if self.model is None:
                input_shape = (X_train.shape[1], X_train.shape[2])
                self.model = self.build_attention_lstm_model(input_shape)
            
            # Entrenar el modelo
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping],
                verbose=1
            )
            
            all_histories.append(history)
            
            # Evaluar el modelo en el conjunto de validación
            y_pred = (self.model.predict(X_val) > 0.5).astype(int)
            
            # Calcular métricas
            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, zero_division=0)
            rec = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            
            logger.info(f"Fold {fold+1} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
            
            fold_metrics.append({
                'fold': fold+1,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1
            })
        
        # Calcular promedios
        avg_metrics = {
            'accuracy': np.mean([m['accuracy'] for m in fold_metrics]),
            'precision': np.mean([m['precision'] for m in fold_metrics]),
            'recall': np.mean([m['recall'] for m in fold_metrics]),
            'f1': np.mean([m['f1'] for m in fold_metrics])
        }
        
        logger.info(f"Validación cruzada completada. Métricas promedio: Accuracy: {avg_metrics['accuracy']:.4f}, " +
                   f"Precision: {avg_metrics['precision']:.4f}, Recall: {avg_metrics['recall']:.4f}, F1: {avg_metrics['f1']:.4f}")
        
        # Guardar historia para visualización
        self.history = all_histories
        
        # Entrenar el modelo final con todos los datos
        logger.info("Entrenando modelo final con todos los datos...")
        final_history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
        return avg_metrics
    
    def train(self, df, epochs=50, batch_size=32, validation_split=0.2):
        """
        Entrena el modelo con los datos proporcionados (método tradicional)
        """
        scaled_data, features, _ = self.prepare_data(df)
        self.num_features = len(features)
        
        X, y = self.create_sequences(scaled_data)
        
        if self.model is None:
            input_shape = (X.shape[1], X.shape[2])
            self.model = self.build_attention_lstm_model(input_shape)
        
        # Configurar Early Stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        
        self.history = history
        logger.info("Entrenamiento completado")
        return history
    
    def predict(self, df):
        """
        Realiza predicciones sobre nuevos datos
        """
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado")
            
        scaled_data, _, _ = self.prepare_data(df)
        
        # Verificar si tenemos suficientes datos para la secuencia
        if len(scaled_data) <= self.sequence_length:
            logger.warning("No hay suficientes datos para hacer una predicción")
            return []
            
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
    
    def plot_training_history(self):
        """
        Visualiza la historia de entrenamiento del modelo
        """
        if self.history is None:
            logger.error("No hay historia de entrenamiento disponible")
            return
        
        # Si es una lista de historias (de validación cruzada)
        if isinstance(self.history, list):
            plt.figure(figsize=(12, 10))
            
            # Graficar pérdida
            plt.subplot(2, 1, 1)
            for i, hist in enumerate(self.history):
                plt.plot(hist.history['loss'], label=f'Fold {i+1} Train')
                plt.plot(hist.history['val_loss'], label=f'Fold {i+1} Val')
            plt.title('Pérdida del Modelo por Folds')
            plt.ylabel('Pérdida')
            plt.xlabel('Epoch')
            plt.legend()
            
            # Graficar precisión
            plt.subplot(2, 1, 2)
            for i, hist in enumerate(self.history):
                plt.plot(hist.history['accuracy'], label=f'Fold {i+1} Train')
                plt.plot(hist.history['val_accuracy'], label=f'Fold {i+1} Val')
            plt.title('Precisión del Modelo por Folds')
            plt.ylabel('Precisión')
            plt.xlabel('Epoch')
            plt.legend()
            
        else:
            # Historia de un solo entrenamiento
            plt.figure(figsize=(12, 5))
            
            # Graficar pérdida
            plt.subplot(1, 2, 1)
            plt.plot(self.history.history['loss'], label='Train')
            plt.plot(self.history.history['val_loss'], label='Validation')
            plt.title('Pérdida del Modelo')
            plt.ylabel('Pérdida')
            plt.xlabel('Epoch')
            plt.legend()
            
            # Graficar precisión
            plt.subplot(1, 2, 2)
            plt.plot(self.history.history['accuracy'], label='Train')
            plt.plot(self.history.history['val_accuracy'], label='Validation')
            plt.title('Precisión del Modelo')
            plt.ylabel('Precisión')
            plt.xlabel('Epoch')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('backtesting_results/model_training_history.png')
        plt.close()
        
        logger.info("Gráfico de historia de entrenamiento guardado en 'backtesting_results/model_training_history.png'") 