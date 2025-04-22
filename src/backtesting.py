import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import pytz
import os
from dotenv import load_dotenv
import logging
from .ml_model import TradingModel
from .trading_bot import TradingBot
import time
import pickle
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.dates as mdates

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BacktestEngine:
    def __init__(self, symbol='EURUSD', timeframe='1h', start_date=None, end_date=None, risk_percent=2.0, stop_loss_pips=50):
        """
        Inicializa el motor de backtesting
        
        Args:
            symbol (str): Símbolo para hacer backtesting
            timeframe (str): Timeframe para los datos ('1m', '5m', '15m', '30m', '1h', '4h', '1d')
            start_date (datetime): Fecha de inicio para el backtesting
            end_date (datetime): Fecha de fin para el backtesting
            risk_percent (float): Porcentaje de riesgo por operación
            stop_loss_pips (int): Stop loss en pips
        """
        self.symbol = symbol
        self.timeframe = timeframe
        
        # Si no se especifican fechas, usar último año
        if start_date is None:
            self.start_date = datetime.now() - timedelta(days=365)
        else:
            self.start_date = start_date
            
        if end_date is None:
            self.end_date = datetime.now()
        else:
            self.end_date = end_date
            
        self.risk_percent = risk_percent
        self.stop_loss_pips = stop_loss_pips
        self.initial_balance = 10000  # Balance inicial para backtest
        self.trades = []
        self.model = TradingModel()
        self.data = None
        self.results = None
        
        # Crear carpeta para resultados
        os.makedirs('backtesting_results', exist_ok=True)
        
    def connect_mt5(self):
        """
        Conecta con MetaTrader 5 para obtener datos históricos
        """
        # Cargar variables de entorno
        load_dotenv()
        
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
            
        logger.info(f"Conectado a MetaTrader5 para backtesting")
        return True
    
    def fetch_historical_data(self):
        """
        Obtiene datos históricos extensos para el backtesting
        """
        # Conectar a MT5 si es necesario
        if not mt5.initialize():
            if not self.connect_mt5():
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
        
        mt5_timeframe = timeframe_dict.get(self.timeframe, mt5.TIMEFRAME_H1)
        
        try:
            # Convertir fechas a formato UTC
            timezone = pytz.timezone("UTC")
            utc_from = timezone.localize(self.start_date)
            utc_to = timezone.localize(self.end_date)
            
            # Obtener datos históricos
            logger.info(f"Obteniendo datos históricos para {self.symbol} desde {self.start_date} hasta {self.end_date}")
            rates = mt5.copy_rates_range(self.symbol, mt5_timeframe, utc_from, utc_to)
            
            if rates is None or len(rates) == 0:
                logger.error(f"Error al obtener datos históricos: {mt5.last_error()}")
                return None
                
            # Convertir a DataFrame
            df = pd.DataFrame(rates)
            df['datetime'] = pd.to_datetime(df['time'], unit='s')
            df = df.set_index('datetime')
            df = df.drop('time', axis=1)
            df = df.rename(columns={'tick_volume': 'volume'})
            
            logger.info(f"Datos históricos obtenidos: {len(df)} registros")
            
            # Guardar datos para referencia
            df.to_csv(f'backtesting_results/{self.symbol}_{self.timeframe}_data.csv')
            
            self.data = df
            return df
            
        except Exception as e:
            logger.error(f"Error al obtener datos históricos: {str(e)}")
            return None
        finally:
            mt5.shutdown()
    
    def prepare_model(self):
        """
        Prepara y entrena el modelo con datos de entrenamiento
        """
        if self.data is None:
            logger.error("No hay datos disponibles para entrenar el modelo")
            return False
            
        # Dividir datos en entrenamiento y prueba (80/20)
        train_size = int(len(self.data) * 0.8)
        train_data = self.data.iloc[:train_size]
        
        logger.info(f"Entrenando modelo con {len(train_data)} registros")
        
        try:
            # Entrenar modelo
            self.model.train(train_data, epochs=20)
            
            # Guardar modelo entrenado
            model_path = Path('backtesting_results/trained_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
                
            logger.info(f"Modelo entrenado y guardado en {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error al entrenar el modelo: {str(e)}")
            return False
    
    def run_backtest(self):
        """
        Ejecuta el backtesting en los datos históricos
        """
        if self.data is None:
            logger.error("No hay datos disponibles para el backtesting")
            return None
            
        # Dividir datos en entrenamiento y prueba (80/20)
        train_size = int(len(self.data) * 0.8)
        test_data = self.data.iloc[train_size:]
        
        logger.info(f"Ejecutando backtesting en {len(test_data)} registros")
        
        # Preparar datos para el seguimiento
        balance = self.initial_balance
        equity = []
        open_positions = []
        trades = []
        
        # Obtener información del símbolo
        point_value = 0.0001 if 'JPY' not in self.symbol else 0.01
        
        # Utilizar la ventana móvil para simular trading en tiempo real
        sequence_length = self.model.sequence_length
        for i in range(sequence_length, len(test_data) - 1):
            try:
                # Crear secuencia de datos hasta el punto actual
                current_window = test_data.iloc[i-sequence_length:i+1]
                
                # Hacer predicción
                prediction = self.model.predict(current_window)
                if len(prediction) == 0:
                    continue
                    
                latest_prediction = prediction[-1][0]
                decision = self.model.evaluate_trading_decision(latest_prediction)
                
                # Datos de precios actuales
                current_price = current_window.iloc[-1]['close']
                next_price = test_data.iloc[i+1]['close']
                timestamp = test_data.index[i]
                
                # Gestionar posiciones abiertas
                for pos in list(open_positions):
                    # Verificar si se activó stop loss o take profit
                    if pos['type'] == 'LONG':
                        # Stop loss
                        sl_price = pos['entry_price'] - (self.stop_loss_pips * point_value)
                        if current_price <= sl_price:
                            profit_loss = (sl_price - pos['entry_price']) * pos['lot_size'] * 100000
                            balance += profit_loss
                            pos['exit_price'] = sl_price
                            pos['exit_time'] = timestamp
                            pos['profit_loss'] = profit_loss
                            pos['exit_reason'] = 'SL'
                            trades.append(pos)
                            open_positions.remove(pos)
                            continue
                            
                        # Take profit (2:1)
                        tp_price = pos['entry_price'] + (self.stop_loss_pips * point_value * 2)
                        if current_price >= tp_price:
                            profit_loss = (tp_price - pos['entry_price']) * pos['lot_size'] * 100000
                            balance += profit_loss
                            pos['exit_price'] = tp_price
                            pos['exit_time'] = timestamp
                            pos['profit_loss'] = profit_loss
                            pos['exit_reason'] = 'TP'
                            trades.append(pos)
                            open_positions.remove(pos)
                            continue
                            
                    elif pos['type'] == 'SHORT':
                        # Stop loss
                        sl_price = pos['entry_price'] + (self.stop_loss_pips * point_value)
                        if current_price >= sl_price:
                            profit_loss = (pos['entry_price'] - sl_price) * pos['lot_size'] * 100000
                            balance += profit_loss
                            pos['exit_price'] = sl_price
                            pos['exit_time'] = timestamp
                            pos['profit_loss'] = profit_loss
                            pos['exit_reason'] = 'SL'
                            trades.append(pos)
                            open_positions.remove(pos)
                            continue
                            
                        # Take profit (2:1)
                        tp_price = pos['entry_price'] - (self.stop_loss_pips * point_value * 2)
                        if current_price <= tp_price:
                            profit_loss = (pos['entry_price'] - tp_price) * pos['lot_size'] * 100000
                            balance += profit_loss
                            pos['exit_price'] = tp_price
                            pos['exit_time'] = timestamp
                            pos['profit_loss'] = profit_loss
                            pos['exit_reason'] = 'TP'
                            trades.append(pos)
                            open_positions.remove(pos)
                            continue
                
                # Cerrar posiciones si hay señal contraria
                for pos in list(open_positions):
                    if (decision == "COMPRA" and pos['type'] == 'SHORT') or (decision == "VENTA" and pos['type'] == 'LONG'):
                        profit_loss = 0
                        if pos['type'] == 'LONG':
                            profit_loss = (current_price - pos['entry_price']) * pos['lot_size'] * 100000
                        else:
                            profit_loss = (pos['entry_price'] - current_price) * pos['lot_size'] * 100000
                            
                        balance += profit_loss
                        pos['exit_price'] = current_price
                        pos['exit_time'] = timestamp
                        pos['profit_loss'] = profit_loss
                        pos['exit_reason'] = 'SIGNAL'
                        trades.append(pos)
                        open_positions.remove(pos)
                
                # Abrir nuevas posiciones
                if decision != "MANTENER":
                    # Calcular tamaño de lote basado en riesgo y balance
                    risk_amount = balance * (self.risk_percent / 100)
                    lot_size = risk_amount / (self.stop_loss_pips * point_value * 10000)
                    
                    # Limitar tamaño de lote
                    lot_size = max(min(lot_size, 1.0), 0.01)
                    lot_size = round(lot_size, 2)  # Redondear a 2 decimales
                    
                    # Crear nueva posición
                    position = {
                        'type': 'LONG' if decision == "COMPRA" else 'SHORT',
                        'entry_price': current_price,
                        'entry_time': timestamp,
                        'lot_size': lot_size,
                        'sl_price': current_price - (self.stop_loss_pips * point_value) if decision == "COMPRA" else current_price + (self.stop_loss_pips * point_value),
                        'tp_price': current_price + (self.stop_loss_pips * point_value * 2) if decision == "COMPRA" else current_price - (self.stop_loss_pips * point_value * 2)
                    }
                    open_positions.append(position)
                
                # Calcular equidad actual (balance + valor de posiciones abiertas)
                current_equity = balance
                for pos in open_positions:
                    if pos['type'] == 'LONG':
                        current_equity += (current_price - pos['entry_price']) * pos['lot_size'] * 100000
                    else:
                        current_equity += (pos['entry_price'] - current_price) * pos['lot_size'] * 100000
                
                equity.append({'timestamp': timestamp, 'equity': current_equity})
                
            except Exception as e:
                logger.error(f"Error en iteración {i}: {str(e)}")
        
        # Cerrar posiciones abiertas al final del backtest
        final_price = test_data.iloc[-1]['close']
        final_time = test_data.index[-1]
        
        for pos in open_positions:
            profit_loss = 0
            if pos['type'] == 'LONG':
                profit_loss = (final_price - pos['entry_price']) * pos['lot_size'] * 100000
            else:
                profit_loss = (pos['entry_price'] - final_price) * pos['lot_size'] * 100000
                
            balance += profit_loss
            pos['exit_price'] = final_price
            pos['exit_time'] = final_time
            pos['profit_loss'] = profit_loss
            pos['exit_reason'] = 'END'
            trades.append(pos)
        
        # Convertir resultados a DataFrame
        trades_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame(equity)
        
        # Guardar resultados
        trades_df.to_csv('backtesting_results/trades.csv', index=False)
        equity_df.to_csv('backtesting_results/equity.csv', index=False)
        
        self.trades = trades_df
        self.results = {
            'equity': equity_df,
            'initial_balance': self.initial_balance,
            'final_balance': balance,
            'profit_loss': balance - self.initial_balance,
            'profit_percent': ((balance / self.initial_balance) - 1) * 100,
            'num_trades': len(trades_df),
            'win_rate': len(trades_df[trades_df['profit_loss'] > 0]) / len(trades_df) * 100 if len(trades_df) > 0 else 0
        }
        
        logger.info(f"Backtesting completado. Ganancia/Pérdida: {self.results['profit_loss']:.2f} ({self.results['profit_percent']:.2f}%)")
        return self.results
    
    def calculate_performance_metrics(self):
        """
        Calcula métricas de rendimiento adicionales
        """
        if self.trades is None or len(self.trades) == 0:
            logger.error("No hay operaciones para calcular métricas")
            return None
            
        if 'equity' not in self.results:
            logger.error("No hay datos de equidad para calcular métricas")
            return None
        
        equity = self.results['equity']
        
        # Convertir a lista de valores
        equity_values = equity['equity'].values
        
        # Calcular drawdown
        peak = equity_values[0]
        drawdown = []
        max_drawdown = 0
        max_drawdown_pct = 0
        
        for eq in equity_values:
            peak = max(peak, eq)
            dd = peak - eq
            dd_pct = (dd / peak) * 100
            drawdown.append(dd_pct)
            
            if dd_pct > max_drawdown_pct:
                max_drawdown = dd
                max_drawdown_pct = dd_pct
        
        # Calcular retornos diarios
        equity['timestamp'] = pd.to_datetime(equity['timestamp'])
        equity = equity.set_index('timestamp')
        daily_returns = equity['equity'].resample('D').last().pct_change().dropna()
        
        # Sharpe Ratio (asumiendo tasa libre de riesgo 0)
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if len(daily_returns) > 0 else 0
        
        # Profit Factor
        winning_trades = self.trades[self.trades['profit_loss'] > 0]
        losing_trades = self.trades[self.trades['profit_loss'] < 0]
        
        gross_profit = winning_trades['profit_loss'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['profit_loss'].sum()) if len(losing_trades) > 0 else 0
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Expectancy
        avg_win = winning_trades['profit_loss'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['profit_loss'].mean() if len(losing_trades) > 0 else 0
        win_rate = len(winning_trades) / len(self.trades) if len(self.trades) > 0 else 0
        
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        # Actualizar resultados
        self.results.update({
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_rate': win_rate * 100
        })
        
        # Mostrar resultados
        logger.info(f"Métricas de rendimiento calculadas:")
        logger.info(f"Ratio de Sharpe: {sharpe_ratio:.2f}")
        logger.info(f"Drawdown máximo: {max_drawdown_pct:.2f}%")
        logger.info(f"Factor de beneficio: {profit_factor:.2f}")
        logger.info(f"Expectativa: {expectancy:.2f}")
        logger.info(f"Tasa de victorias: {win_rate * 100:.2f}%")
        
        return self.results
    
    def generate_report(self):
        """
        Genera un informe visual del backtesting
        """
        if self.results is None:
            logger.error("No hay resultados para generar informe")
            return
            
        # Configurar estilo de gráficos
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("viridis")
        
        # Crear figura
        fig = plt.figure(figsize=(15, 20))
        
        # 1. Curva de equidad
        ax1 = fig.add_subplot(4, 1, 1)
        equity_df = self.results['equity']
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        
        ax1.plot(equity_df['timestamp'], equity_df['equity'], linewidth=2)
        ax1.set_title('Curva de Equidad', fontsize=14)
        ax1.set_ylabel('Equidad ($)', fontsize=12)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45)
        
        # 2. Distribución de ganancias/pérdidas
        ax2 = fig.add_subplot(4, 1, 2)
        sns.histplot(self.trades['profit_loss'], bins=30, kde=True, ax=ax2)
        ax2.set_title('Distribución de Ganancias/Pérdidas', fontsize=14)
        ax2.set_xlabel('Ganancia/Pérdida ($)', fontsize=12)
        ax2.set_ylabel('Frecuencia', fontsize=12)
        
        # 3. Operaciones a lo largo del tiempo
        ax3 = fig.add_subplot(4, 1, 3)
        
        # Convertir columnas de tiempo
        self.trades['entry_time'] = pd.to_datetime(self.trades['entry_time'])
        self.trades['exit_time'] = pd.to_datetime(self.trades['exit_time'])
        
        # Graficar operaciones ganadoras en verde, perdedoras en rojo
        winning_trades = self.trades[self.trades['profit_loss'] > 0]
        losing_trades = self.trades[self.trades['profit_loss'] < 0]
        
        ax3.scatter(winning_trades['entry_time'], winning_trades['profit_loss'], c='green', label='Ganadora', alpha=0.7)
        ax3.scatter(losing_trades['entry_time'], losing_trades['profit_loss'], c='red', label='Perdedora', alpha=0.7)
        
        ax3.set_title('Operaciones a lo largo del tiempo', fontsize=14)
        ax3.set_ylabel('Ganancia/Pérdida ($)', fontsize=12)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.legend()
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45)
        
        # 4. Tabla de resumen
        ax4 = fig.add_subplot(4, 1, 4)
        ax4.axis('off')
        
        metrics = [
            f"Balance Inicial: ${self.results['initial_balance']:.2f}",
            f"Balance Final: ${self.results['final_balance']:.2f}",
            f"Ganancia Total: ${self.results['profit_loss']:.2f} ({self.results['profit_percent']:.2f}%)",
            f"Número de Operaciones: {self.results['num_trades']}",
            f"Tasa de Victorias: {self.results['win_rate']:.2f}%",
            f"Ratio de Sharpe: {self.results['sharpe_ratio']:.2f}",
            f"Drawdown Máximo: {self.results['max_drawdown_pct']:.2f}%",
            f"Factor de Beneficio: {self.results['profit_factor']:.2f}",
            f"Expectativa: ${self.results['expectancy']:.2f}",
            f"Ganancia Media: ${self.results['avg_win']:.2f}",
            f"Pérdida Media: ${self.results['avg_loss']:.2f}"
        ]
        
        y_pos = 0.9
        for metric in metrics:
            ax4.text(0.5, y_pos, metric, ha='center', fontsize=12)
            y_pos -= 0.08
        
        # Ajustar diseño y guardar
        plt.tight_layout()
        plt.savefig('backtesting_results/backtest_report.png', dpi=300)
        plt.savefig('backtesting_results/backtest_report.pdf')
        
        logger.info(f"Informe generado y guardado en backtesting_results/")
        
        # Generar archivo de resumen en texto
        with open('backtesting_results/backtest_summary.txt', 'w') as f:
            f.write(f"Informe de Backtesting para {self.symbol} en {self.timeframe}\n")
            f.write(f"Período: {self.start_date.strftime('%Y-%m-%d')} a {self.end_date.strftime('%Y-%m-%d')}\n\n")
            
            for metric in metrics:
                f.write(f"{metric}\n")
                
        return fig 