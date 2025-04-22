from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import torch
import json

def cargar_modelo():
    # Cargar el modelo preentrenado de FinBERT
    modelo = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
    analizador_sentimientos = pipeline('sentiment-analysis', model=modelo, tokenizer=tokenizer)
    return analizador_sentimientos

def analizar_sentimiento(texto):
    analizador = cargar_modelo()
    resultado = analizador(texto)
    return resultado

def analizar_noticias(file_path):
    # Leer las noticias procesadas y analizar el sentimiento
    with open(file_path, 'r', encoding='utf-8') as file:
        noticias = json.load(file)

    noticias_analizadas = []
    for articulo in noticias['articles']:
        titulo = articulo['title']
        contenido = articulo['content']
        sentimiento = analizar_sentimiento(titulo + " " + contenido)
        noticia_analizada = {
            "title": titulo,
            "sentiment": sentimiento[0]['label'],
            "score": sentimiento[0]['score']
        }
        noticias_analizadas.append(noticia_analizada)

    # Guardar los resultados en un archivo
    archivo_sentimiento = f"news/sentiment/sentiment_{file_path.split('/')[-1]}"
    with open(archivo_sentimiento, 'w', encoding='utf-8') as out_file:
        json.dump(noticias_analizadas, out_file, ensure_ascii=False, indent=4)

    print(f"Análisis de sentimiento realizado y guardado en {archivo_sentimiento}")
