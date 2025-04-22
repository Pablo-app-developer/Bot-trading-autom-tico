import logging
from src.trading_bot import TradingBot
from dotenv import load_dotenv
import os
import sys

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
            symbol='EURUSD'  # Par de divisas
        )
        
        # Verificar si se pasó un tamaño de lote como argumento
        lot_size = 0.01  # Valor predeterminado
        if len(sys.argv) > 1:
            try:
                lot_size = float(sys.argv[1])
                logger.info(f"Usando tamaño de lote personalizado: {lot_size}")
            except ValueError:
                logger.warning(f"Valor de lote inválido: {sys.argv[1]}. Usando valor predeterminado: {lot_size}")
        
        # Ejecutar el bot con el tamaño de lote especificado
        try:
            bot.run(lot_size=lot_size)
        except KeyboardInterrupt:
            logger.info("Bot detenido por el usuario")
        finally:
            # Asegurarse de cerrar la conexión con MetaTrader
            bot.shutdown()
        
    except Exception as e:
        logger.error(f"Error en la ejecución del bot: {str(e)}")

if __name__ == "__main__":
    main()
