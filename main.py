import logging
from src.trading_bot import TradingBot
from dotenv import load_dotenv
import os
import sys
import argparse

# Crear directorio de logs si no existe
os.makedirs('logs', exist_ok=True)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_bot.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Bot de Trading para MetaTrader 5')
    parser.add_argument('--symbol', type=str, default='EURUSD', help='Símbolo para operar (por defecto: EURUSD)')
    parser.add_argument('--risk', type=float, default=2.0, help='Porcentaje de riesgo por operación (por defecto: 2.0%%)')
    parser.add_argument('--sl', type=int, default=50, help='Stop loss en pips (por defecto: 50)')
    parser.add_argument('--manual-lot', type=float, help='Tamaño de lote fijo (si no se especifica, se calcula automáticamente)')
    parser.add_argument('--max-lot', type=float, default=1.0, help='Tamaño máximo de lote permitido (por defecto: 1.0)')
    
    args = parser.parse_args()
    
    # Cargar variables de entorno
    load_dotenv()
    
    # Verificar que las credenciales estén configuradas
    if not os.getenv('MT_ACCOUNT') or not os.getenv('MT_PASSWORD') or not os.getenv('MT_SERVER'):
        logger.error("Error: Las credenciales de MetaTrader no están configuradas correctamente en el archivo .env")
        logger.info("Por favor, configura MT_ACCOUNT, MT_PASSWORD y MT_SERVER en el archivo .env")
        return
    
    try:
        # Inicializar y ejecutar el bot
        bot = TradingBot(
            symbol=args.symbol,
            risk_percent=args.risk,
            max_lot_size=args.max_lot
        )
        
        logger.info(f"Iniciando bot con símbolo: {args.symbol}, riesgo: {args.risk}%, stop loss: {args.sl} pips")
        if args.manual_lot:
            logger.info(f"Usando tamaño de lote manual: {args.manual_lot}")
        else:
            logger.info("Calculando tamaño de lote automáticamente en base al riesgo y balance")
        
        # Ejecutar el bot
        try:
            bot.run(lot_size=args.manual_lot, stop_loss_pips=args.sl)
        except KeyboardInterrupt:
            logger.info("Bot detenido por el usuario")
        finally:
            # Asegurarse de cerrar la conexión con MetaTrader
            bot.shutdown()
        
    except Exception as e:
        logger.error(f"Error en la ejecución del bot: {str(e)}")

if __name__ == "__main__":
    main()
