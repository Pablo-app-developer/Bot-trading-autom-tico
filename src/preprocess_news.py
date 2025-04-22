import os
import json

RAW_NEWS_DIR = os.path.join("news", "raw")
CLEAN_NEWS_DIR = os.path.join("news", "clean")

def crear_directorio_salida():
    os.makedirs(CLEAN_NEWS_DIR, exist_ok=True)
    print(f"📁 Carpeta de salida asegurada: {CLEAN_NEWS_DIR}")

def obtener_archivos_raw():
    archivos = [f for f in os.listdir(RAW_NEWS_DIR) if f.endswith(".json")]
    if not archivos:
        print("No se encontraron archivos en 'news/raw/'. Asegúrate de haber ejecutado fetch_news.py.")
    else:
        print(f"Archivos encontrados en 'raw': {archivos}")
    return archivos

if __name__ == "__main__":
    crear_directorio_salida()
    obtener_archivos_raw()


from datetime import datetime

import re
from bs4 import BeautifulSoup
import unicodedata

def limpiar_texto(texto):
    if not texto:
        return ""

    # Quitar etiquetas HTML
    texto = BeautifulSoup(texto, "html.parser").get_text()

    # Eliminar URLs
    texto = re.sub(r"http\S+|www\S+|https\S+", "", texto)

    # Normalizar tildes y acentos, pero conservar letras
    texto = unicodedata.normalize("NFKD", texto)
    texto = texto.encode("ASCII", "ignore").decode("utf-8")

    # Eliminar caracteres especiales
    texto = re.sub(r"[^a-zA-Z0-9\s]", "", texto)

    # Convertir a minúsculas
    texto = texto.lower()

    # Eliminar espacios múltiples
    texto = re.sub(r"\s+", " ", texto).strip()

    return texto


def procesar_archivo(nombre_archivo):
    ruta_entrada = os.path.join(RAW_NEWS_DIR, nombre_archivo)
    with open(ruta_entrada, "r", encoding="utf-8") as f:
        datos = json.load(f)

    articulos = datos.get("articles", [])
    noticias_limpias = []

    for articulo in articulos:
        noticia = {
            "titulo": limpiar_texto(articulo.get("title")),
            "descripcion": limpiar_texto(articulo.get("description")),
            "contenido": limpiar_texto(articulo.get("content")),
            "fecha": articulo.get("publishedAt")
        }
        noticias_limpias.append(noticia)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_salida = os.path.join(CLEAN_NEWS_DIR, f"cleaned_news_{timestamp}.json")

    with open(ruta_salida, "w", encoding="utf-8") as f:
        json.dump(noticias_limpias, f, indent=2, ensure_ascii=False)

    print(f"Noticias procesadas y guardadas en: {ruta_salida}")


if __name__ == "__main__":
    crear_directorio_salida()
    archivos = obtener_archivos_raw()
    for archivo in archivos:
        procesar_archivo(archivo)

