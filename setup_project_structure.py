import os

# Carpetas base
folders = [
    "data/external",
    "data/processed",
    "news/raw",
    "news/parsed",
    "models",
    "notebooks",
    "reports/figures",
    "src",
    "tests"
]

# Archivos vacíos a crear
files = {
    "README.md": "",
    ".gitignore": "venv/\n__pycache__/\n.env\n*.pyc\n",
    "requirements.txt": "",
    ".env": "# API_KEY=your_api_key_here\n",
    "src/__init__.py": "",
    "src/config.py": "# Parámetros de configuración global\n",
    "src/fetch_news.py": "# Scraping o API para noticias\n",
    "src/preprocess.py": "# Limpieza de datos y noticias\n",
    "src/sentiment_model.py": "# FinBERT o GPT para análisis de sentimiento\n",
    "src/feature_engineer.py": "# RSI, MACD, etc.\n",
    "src/strategy.py": "# Lógica de decisiones de trading\n",
    "src/train_model.py": "# Entrenamiento del modelo\n",
    "src/bot_core.py": "# Lógica principal del bot\n",
    "tests/test_fetch_data.py": "# Prueba unitaria básica\n",
    "main.py": "# Punto de entrada del bot\n"
}

def create_structure():
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"✅ Carpeta asegurada: {folder}")

    for path, content in files.items():
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
                print(f"📝 Archivo creado: {path}")
        else:
            print(f"⚠️  Archivo ya existe, omitido: {path}")

if __name__ == "__main__":
    create_structure()
