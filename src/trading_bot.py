import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from .ml_model import TradingModel
import os
from dotenv import load_dotenv
import time

# Cargar variables de entorno
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self, exchange_id='binance', symbol='EUR/USD'):
        self.exchange = getattr(ccxt, exchange_id)({
            'apiKey': os.getenv('EXCHANGE_API_KEY'),
            'secret': os.getenv('EXCHANGE_SECRET'),
            'enableRateLimit': True
        })
        self.symbol = symbol
        self.model = TradingModel()
        self.position = None
        self.last_trade_time = None
        self.min_trade_interval = timedelta(minutes=15)
        
    def fetch_historical_data(self, timeframe='1h', limit=1000):
        """
        Obtiene datos históricos del par de trading
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"Error al obtener datos históricos: {str(e)}")
            return None
    
    def train_model(self, df):
        """
        Entrena el modelo con datos históricos
        """
        try:
            history = self.model.train(df)
            logger.info("Modelo entrenado exitosamente")
            return history
        except Exception as e:
            logger.error(f"Error al entrenar el modelo: {str(e)}")
            return None
    
    def get_current_position(self):
        """
        Obtiene la posición actual en el mercado
        """
        try:
            balance = self.exchange.fetch_balance()
            return balance['total']
        except Exception as e:
            logger.error(f"Error al obtener el balance: {str(e)}")
            return None
    
    def execute_trade(self, decision, amount):
        """
        Ejecuta una operación de trading
        """
        try:
            if decision == "COMPRA" and self.position != "LONG":
                order = self.exchange.create_market_buy_order(self.symbol, amount)
                self.position = "LONG"
                self.last_trade_time = datetime.now()
                logger.info(f"Orden de compra ejecutada: {order}")
            elif decision == "VENTA" and self.position != "SHORT":
                order = self.exchange.create_market_sell_order(self.symbol, amount)
                self.position = "SHORT"
                self.last_trade_time = datetime.now()
                logger.info(f"Orden de venta ejecutada: {order}")
        except Exception as e:
            logger.error(f"Error al ejecutar la operación: {str(e)}")
    
    def run(self, trade_amount=0.01):
        """
        Ejecuta el bot de trading
        """
        logger.info("Iniciando bot de trading...")
        
        # Obtener datos históricos y entrenar el modelo
        historical_data = self.fetch_historical_data()
        if historical_data is not None:
            self.train_model(historical_data)
        
        while True:
            try:
                # Verificar si ha pasado suficiente tiempo desde la última operación
                if self.last_trade_time and datetime.now() - self.last_trade_time < self.min_trade_interval:
                    logger.info("Esperando intervalo mínimo entre operaciones...")
                    continue
                
                # Obtener datos actuales
                current_data = self.fetch_historical_data(limit=100)
                if current_data is None:
                    continue
                
                # Realizar predicción
                predictions = self.model.predict(current_data)
                latest_prediction = predictions[-1][0]
                
                # Evaluar decisión de trading
                decision = self.model.evaluate_trading_decision(latest_prediction)
                logger.info(f"Decisión de trading: {decision}")
                
                # Ejecutar operación si es necesario
                if decision != "MANTENER":
                    self.execute_trade(decision, trade_amount)
                
                # Esperar antes de la siguiente iteración
                time.sleep(60)  # Esperar 1 minuto
                
            except Exception as e:
                logger.error(f"Error en el ciclo principal: {str(e)}")
                time.sleep(60)  # Esperar 1 minuto antes de reintentar 