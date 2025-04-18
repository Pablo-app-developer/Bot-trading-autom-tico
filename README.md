
# 🤖 Bot de Trading Automático EUR/USD

Este proyecto implementa un sistema profesional de trading automático para el par de divisas **EUR/USD**, integrando análisis de noticias, procesamiento de datos y futura toma de decisiones mediante inteligencia artificial.

---

## 📁 Estructura del Proyecto

```
eurusd_bot/
├── data/                  # Datos financieros históricos (precios, indicadores, etc.)
├── news/
│   └── raw/               # Archivos JSON con noticias financieras crudas
├── models/                # Modelos entrenados (NLP, predicción, RL)
├── src/                   # Scripts fuente principales
│   ├── fetch_data.py      # Obtención de datos de precios históricos
│   ├── fetch_news.py      # Descarga de noticias financieras vía API
│   ├── preprocess_news.py # Limpieza y tokenización de textos noticiosos
│   └── ...                # Otros scripts (training, bot, análisis)
├── .env                   # Variables sensibles (NO subir al repo)
├── .gitignore             # Ignora carpetas temporales y archivos sensibles
├── requirements.txt       # Dependencias del proyecto
└── README.md              # Documentación general (este archivo)
```

---

## ⚙️ Requisitos

- Python 3.11+
- Entorno virtual (`venv`)
- Acceso a [https://newsapi.org](https://newsapi.org) con API Key
- Git instalado y configurado

---

## 🚀 Cómo ejecutar

1. Clona el repositorio:

```bash
git clone https://github.com/Pablo-app-developer/Bot-trading-autom-tico.git
cd eurusd_bot
```

2. Crea y activa el entorno virtual:

```bash
python -m venv venv
.env\Scriptsctivate
```

3. Instala las dependencias:

```bash
pip install -r requirements.txt
```

4. Crea un archivo `.env` y añade tu clave de NewsAPI:

```env
NEWS_API_KEY=tu_clave_api_aqui
```

5. Ejecuta el script para descargar noticias:

```bash
python src/fetch_news.py
```

---

## 🧠 Fases del proyecto

| Fase | Descripción |
|------|-------------|
| ✅ 1 | Configuración del entorno y estructura del proyecto |
| ✅ 2 | Descarga de datos históricos y noticias recientes |
| 🔄 3 | Preprocesamiento de noticias para NLP |
| 🔜 4 | Análisis de sentimiento con FinBERT u otro modelo |
| 🔜 5 | Entrenamiento de estrategia de trading con IA (ej. RL) |
| 🔜 6 | Integración con plataforma de trading (OANDA, MetaTrader) |
| 🔜 7 | Interfaz visual (Gradio o Streamlit) |

---

## 📌 Notas importantes

- Este proyecto está diseñado para fines educativos y de investigación.
- No se recomienda su uso directo en cuentas reales sin validación profesional y auditoría de riesgos.

---

## 👨‍💻 Autor

**Pablo Ramírez**  
Ingeniero Industrial, entusiasta del trading algorítmico y la programación en Python.

---

