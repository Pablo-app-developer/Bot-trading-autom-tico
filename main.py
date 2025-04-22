import logging
from src.trading_bot import TradingBot
from dotenv import load_dotenv
import os

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
    if not os.getenv('EXCHANGE_API_KEY') or not os.getenv('EXCHANGE_SECRET'):
        logger.error("Error: Las credenciales de la API no están configuradas")
        return
    
    try:
        # Inicializar y ejecutar el bot
        bot = TradingBot(
            exchange_id='binance',  # Puedes cambiar esto según el exchange que uses
            symbol='EUR/USD'
        )
        
        # Ejecutar el bot con un tamaño de operación de 0.01 lotes
        bot.run(trade_amount=0.01)
        
    except Exception as e:
        logger.error(f"Error en la ejecución del bot: {str(e)}")

if __name__ == "__main__":
    main()
