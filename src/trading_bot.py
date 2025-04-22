import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from .ml_model import TradingModel
import os
from dotenv import load_dotenv
import time
import pytz

# Cargar variables de entorno
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self, symbol='EURUSD', risk_percent=2.0, max_lot_size=1.0):
        self.symbol = symbol
        self.model = TradingModel()
        self.position = None
        self.last_trade_time = None
        self.min_trade_interval = timedelta(minutes=15)
        self.initialized = False
        self.risk_percent = risk_percent  # Porcentaje de la cuenta a arriesgar por operación
        self.max_lot_size = max_lot_size  # Tamaño máximo de lote permitido
        
    def initialize(self):
        """
        Inicializa la conexión con MetaTrader 5
        """
        if not mt5.initialize():
            logger.error(f"Error al inicializar MetaTrader5: {mt5.last_error()}")
            return False
            
        # Configurar la cuenta
        account = os.getenv('MT_ACCOUNT')
        password = os.getenv('MT_PASSWORD')
        server = os.getenv('MT_SERVER')
        
        if not mt5.login(int(account), password=password, server=server):
            logger.error(f"Error al iniciar sesión en MetaTrader5: {mt5.last_error()}")
            mt5.shutdown()
            return False
            
        logger.info(f"Conectado a MetaTrader5: {mt5.terminal_info()}")
        logger.info(f"Cuenta: {mt5.account_info()}")
        
        # Verificar que el símbolo existe
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            logger.error(f"El símbolo {self.symbol} no está disponible")
            mt5.shutdown()
            return False
            
        if not symbol_info.visible:
            logger.info(f"Habilitando símbolo {self.symbol}")
            if not mt5.symbol_select(self.symbol, True):
                logger.error(f"Error al habilitar el símbolo {self.symbol}")
                mt5.shutdown()
                return False
                
        self.initialized = True
        return True
        
    def fetch_historical_data(self, timeframe='1h', limit=1000):
        """
        Obtiene datos históricos del par de trading desde MetaTrader
        """
        if not self.initialized and not self.initialize():
            return None
            
        # Convertir timeframe string a constante MT5
        timeframe_dict = {
            '1m': mt5.TIMEFRAME_M1,
            '5m': mt5.TIMEFRAME_M5,
            '15m': mt5.TIMEFRAME_M15,
            '30m': mt5.TIMEFRAME_M30,
            '1h': mt5.TIMEFRAME_H1,
            '4h': mt5.TIMEFRAME_H4,
            '1d': mt5.TIMEFRAME_D1
        }
        
        mt5_timeframe = timeframe_dict.get(timeframe, mt5.TIMEFRAME_H1)
        
        try:
            # Obtener zona horaria UTC para timestamps
            timezone = pytz.timezone("UTC")
            utc_from = datetime.now(tz=timezone) - timedelta(days=limit/24) if timeframe == '1h' else datetime.now(tz=timezone) - timedelta(days=limit/24/60)
            
            # Obtener datos históricos
            rates = mt5.copy_rates_from(self.symbol, mt5_timeframe, utc_from, limit)
            
            if rates is None or len(rates) == 0:
                logger.error(f"Error al obtener datos históricos: {mt5.last_error()}")
                return None
                
            # Convertir a DataFrame
            df = pd.DataFrame(rates)
            df['timestamp'] = pd.to_datetime(df['time'], unit='s')
            df = df.drop('time', axis=1)
            df = df.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'tick_volume': 'volume'})
            
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
        if not self.initialized and not self.initialize():
            return None
            
        try:
            positions = mt5.positions_get(symbol=self.symbol)
            
            if positions is None:
                logger.error(f"Error al obtener posiciones: {mt5.last_error()}")
                return None
                
            if len(positions) == 0:
                return "FLAT"  # Sin posición
                
            # Determinar tipo de posición
            long_volume = sum(pos.volume for pos in positions if pos.type == mt5.POSITION_TYPE_BUY)
            short_volume = sum(pos.volume for pos in positions if pos.type == mt5.POSITION_TYPE_SELL)
            
            if long_volume > short_volume:
                return "LONG"
            elif short_volume > long_volume:
                return "SHORT"
            else:
                return "FLAT"
                
        except Exception as e:
            logger.error(f"Error al obtener el balance: {str(e)}")
            return None
    
    def calculate_lot_size(self, stop_loss_pips=50):
        """
        Calcula el tamaño del lote basado en el balance de la cuenta, el riesgo y el stop loss
        
        Args:
            stop_loss_pips (int): Distancia del stop loss en pips
            
        Returns:
            float: Tamaño del lote calculado
        """
        if not self.initialized and not self.initialize():
            return 0.01  # Valor mínimo por defecto
            
        try:
            # Obtener información de la cuenta
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("No se pudo obtener información de la cuenta")
                return 0.01
                
            # Obtener información del símbolo
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                logger.error(f"No se pudo obtener información del símbolo {self.symbol}")
                return 0.01
                
            # Obtener el balance y calcular la cantidad a arriesgar
            balance = account_info.balance
            risk_amount = balance * (self.risk_percent / 100)
            
            # Obtener el valor del pip
            # Para la mayoría de pares, 1 pip = 0.0001, excepto JPY donde 1 pip = 0.01
            digit_multiplier = 10000 if symbol_info.digits in [4, 2] else 100
            pip_value = symbol_info.trade_tick_value * (digit_multiplier / symbol_info.point)
            
            # Calcular el valor de un pip en la moneda de la cuenta
            one_pip_value = pip_value / digit_multiplier
            
            # Calcular el tamaño del lote basado en el riesgo y el stop loss
            lot_size = risk_amount / (stop_loss_pips * one_pip_value)
            
            # Ajustar al tamaño de lote mínimo y máximo
            min_lot = symbol_info.volume_min
            max_lot = min(symbol_info.volume_max, self.max_lot_size)
            step = symbol_info.volume_step
            
            # Redondear al step más cercano
            lot_size = round(max(min(lot_size, max_lot), min_lot) / step) * step
            
            logger.info(f"Balance: {balance}, Riesgo: {risk_amount}, Stop Loss: {stop_loss_pips} pips")
            logger.info(f"Tamaño de lote calculado: {lot_size}")
            
            return lot_size
            
        except Exception as e:
            logger.error(f"Error al calcular el tamaño del lote: {str(e)}")
            return 0.01  # Valor predeterminado seguro
    
    def execute_trade(self, decision, lot_size=None, stop_loss_pips=50):
        """
        Ejecuta una operación de trading en MetaTrader
        """
        if not self.initialized and not self.initialize():
            return
            
        try:
            # Si no se proporciona un tamaño de lote, calcularlo automáticamente
            if lot_size is None:
                lot_size = self.calculate_lot_size(stop_loss_pips)
                
            # Cerrar posiciones existentes primero si cambiamos de dirección
            current_position = self.get_current_position()
            
            if (decision == "COMPRA" and current_position == "SHORT") or \
               (decision == "VENTA" and current_position == "LONG"):
                self.close_all_positions()
                
            # Obtener información del símbolo
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                logger.error(f"Error al obtener información del símbolo: {self.symbol}")
                return
                
            # Preparar solicitud de operación
            point = symbol_info.point
            price = mt5.symbol_info_tick(self.symbol).ask if decision == "COMPRA" else mt5.symbol_info_tick(self.symbol).bid
            deviation = 20  # Desviación en puntos
            
            # Calcular niveles de stop loss y take profit si se desea
            sl_price = 0
            tp_price = 0
            
            if stop_loss_pips > 0:
                sl_price = price - stop_loss_pips * point * 10 if decision == "COMPRA" else price + stop_loss_pips * point * 10
                # 2:1 risk/reward ratio para take profit
                tp_price = price + (stop_loss_pips * 2) * point * 10 if decision == "COMPRA" else price - (stop_loss_pips * 2) * point * 10
            
            # Configuración de la solicitud
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": lot_size,
                "type": mt5.ORDER_TYPE_BUY if decision == "COMPRA" else mt5.ORDER_TYPE_SELL,
                "price": price,
                "sl": sl_price,
                "tp": tp_price,
                "deviation": deviation,
                "magic": 12345,  # ID para identificar operaciones del bot
                "comment": "Bot Trading Operación",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Enviar orden
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Error al ejecutar la operación: {result.retcode}, {result.comment}")
                return
                
            logger.info(f"Orden de {'compra' if decision == 'COMPRA' else 'venta'} ejecutada: {result.order}, Lote: {lot_size}")
            self.position = "LONG" if decision == "COMPRA" else "SHORT"
            self.last_trade_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Error al ejecutar la operación: {str(e)}")
    
    def close_all_positions(self):
        """
        Cierra todas las posiciones abiertas para el símbolo
        """
        if not self.initialized and not self.initialize():
            return
            
        try:
            positions = mt5.positions_get(symbol=self.symbol)
            
            if positions is None:
                logger.error(f"Error al obtener posiciones para cerrar: {mt5.last_error()}")
                return
                
            for position in positions:
                # Crear solicitud para cerrar posición
                tick = mt5.symbol_info_tick(self.symbol)
                price = tick.bid if position.type == mt5.POSITION_TYPE_BUY else tick.ask
                
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": self.symbol,
                    "volume": position.volume,
                    "type": mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                    "position": position.ticket,
                    "price": price,
                    "deviation": 20,
                    "magic": 12345,
                    "comment": "Bot Trading Cierre",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                
                result = mt5.order_send(request)
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    logger.error(f"Error al cerrar posición: {result.retcode}, {result.comment}")
                else:
                    logger.info(f"Posición cerrada: {position.ticket}")
                    
        except Exception as e:
            logger.error(f"Error al cerrar posiciones: {str(e)}")
    
    def run(self, lot_size=None, risk_percent=None, stop_loss_pips=50):
        """
        Ejecuta el bot de trading
        """
        logger.info("Iniciando bot de trading para MetaTrader 5...")
        
        # Actualizar el porcentaje de riesgo si se proporciona
        if risk_percent is not None:
            self.risk_percent = risk_percent
            logger.info(f"Porcentaje de riesgo configurado a {self.risk_percent}%")
        
        # Inicializar conexión a MetaTrader
        if not self.initialize():
            logger.error("No se pudo inicializar la conexión con MetaTrader 5")
            return
        
        # Obtener datos históricos y entrenar el modelo
        historical_data = self.fetch_historical_data()
        if historical_data is not None:
            self.train_model(historical_data)
        else:
            logger.error("No se pudo obtener datos históricos para entrenar el modelo")
            return
        
        logger.info("Bot listo para operar")
        
        while True:
            try:
                # Verificar si ha pasado suficiente tiempo desde la última operación
                if self.last_trade_time and datetime.now() - self.last_trade_time < self.min_trade_interval:
                    logger.info("Esperando intervalo mínimo entre operaciones...")
                    time.sleep(60)  # Esperar 1 minuto
                    continue
                
                # Obtener datos actuales
                current_data = self.fetch_historical_data(limit=100)
                if current_data is None:
                    logger.error("No se pudieron obtener datos actuales")
                    time.sleep(60)
                    continue
                
                # Realizar predicción
                predictions = self.model.predict(current_data)
                latest_prediction = predictions[-1][0]
                
                # Evaluar decisión de trading
                decision = self.model.evaluate_trading_decision(latest_prediction)
                logger.info(f"Decisión de trading: {decision}")
                
                # Ejecutar operación si es necesario
                if decision != "MANTENER":
                    self.execute_trade(decision, lot_size=lot_size, stop_loss_pips=stop_loss_pips)
                
                # Esperar antes de la siguiente iteración
                time.sleep(60)  # Esperar 1 minuto
                
            except Exception as e:
                logger.error(f"Error en el ciclo principal: {str(e)}")
                time.sleep(60)  # Esperar 1 minuto antes de reintentar
                
    def shutdown(self):
        """
        Cierra la conexión con MetaTrader 5
        """
        if self.initialized:
            mt5.shutdown()
            logger.info("Conexión con MetaTrader 5 cerrada") 