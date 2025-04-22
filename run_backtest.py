import argparse
from datetime import datetime
import logging
from src.backtesting import BacktestEngine
import os
import matplotlib.pyplot as plt

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtesting_results/backtest.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    # Crear directorio para resultados si no existe
    os.makedirs('backtesting_results', exist_ok=True)
    
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Backtesting de Estrategia de Trading')
    parser.add_argument('--symbol', type=str, default='EURUSD', help='Símbolo para backtesting (por defecto: EURUSD)')
    parser.add_argument('--timeframe', type=str, default='1h', help='Timeframe para los datos (por defecto: 1h)')
    parser.add_argument('--start-date', type=str, help='Fecha de inicio en formato YYYY-MM-DD')
    parser.add_argument('--end-date', type=str, help='Fecha de fin en formato YYYY-MM-DD')
    parser.add_argument('--risk', type=float, default=2.0, help='Porcentaje de riesgo por operación (por defecto: 2.0%%)')
    parser.add_argument('--sl', type=int, default=50, help='Stop loss en pips (por defecto: 50)')
    parser.add_argument('--no-plots', action='store_true', help='No generar gráficos')
    
    args = parser.parse_args()
    
    # Convertir fechas si se proporcionan
    start_date = None
    end_date = None
    
    if args.start_date:
        try:
            start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        except ValueError:
            logger.error(f"Formato de fecha de inicio inválido. Use YYYY-MM-DD")
            return
    
    if args.end_date:
        try:
            end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
        except ValueError:
            logger.error(f"Formato de fecha de fin inválido. Use YYYY-MM-DD")
            return
    
    logger.info(f"Iniciando backtesting para {args.symbol} en timeframe {args.timeframe}")
    
    # Inicializar motor de backtesting
    backtest = BacktestEngine(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=start_date,
        end_date=end_date,
        risk_percent=args.risk,
        stop_loss_pips=args.sl
    )
    
    # Ejecutar backtesting
    try:
        # Obtener datos históricos
        logger.info("Obteniendo datos históricos...")
        if backtest.fetch_historical_data() is None:
            logger.error("No se pudieron obtener datos históricos. Abortando backtesting.")
            return
            
        # Preparar y entrenar modelo
        logger.info("Entrenando modelo...")
        if not backtest.prepare_model():
            logger.error("Error al entrenar el modelo. Abortando backtesting.")
            return
            
        # Ejecutar backtesting
        logger.info("Ejecutando simulación de backtesting...")
        results = backtest.run_backtest()
        
        if results is None:
            logger.error("Error durante el backtesting")
            return
            
        # Calcular métricas de rendimiento
        logger.info("Calculando métricas de rendimiento...")
        backtest.calculate_performance_metrics()
        
        # Generar informe
        if not args.no_plots:
            logger.info("Generando informe visual...")
            fig = backtest.generate_report()
            
            # Mostrar gráfico si no estamos en modo headless
            plt.show()
            
        logger.info("Backtesting completado con éxito")
        logger.info(f"Ganancia total: ${results['profit_loss']:.2f} ({results['profit_percent']:.2f}%)")
        logger.info(f"Resultados guardados en directorio 'backtesting_results/'")
            
    except Exception as e:
        logger.error(f"Error durante el backtesting: {str(e)}")
        return

if __name__ == "__main__":
    main() 