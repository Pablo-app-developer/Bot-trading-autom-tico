# Scraping o API para noticias


import os
import requests
from dotenv import load_dotenv
import json
from datetime import datetime, timedelta

load_dotenv()

API_KEY = os.getenv("NEWS_API_KEY")
NEWS_ENDPOINT = "https://newsapi.org/v2/everything"

def fetch_news(query="EUR/USD", from_date=None, to_date=None, language="en"):
    if not from_date:
        from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    if not to_date:
        to_date = datetime.now().strftime("%Y-%m-%d")
    params = {
        "q": query,
        "from": from_date,
        "to": to_date,
        "sortBy": "relevancy",
        "language": language,
        "apiKey": API_KEY,
        "pageSize": 100  # Máximo por petición
    }

    response = requests.get(NEWS_ENDPOINT, params=params)
    data = response.json()

    # Validar respuesta
    if response.status_code != 200:
        raise Exception(f"Error al obtener noticias: {data.get('message')}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"news/raw/news_{timestamp}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ {len(data['articles'])} noticias guardadas en {output_file}")
    return data



# Para pruebas rápidas desde consola
if __name__ == "__main__":
    fetch_news()

load_dotenv()
API_KEY = os.getenv("NEWS_API_KEY")

if not API_KEY:
    raise Exception("❌ ERROR: No se encontró la clave API. Verifica tu archivo .env")

print(f"🔑 Clave API cargada: {API_KEY[:5]}...")  # Solo muestra los primeros caracteres

# Continúa con fetch_news...
