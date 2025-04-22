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
- MetaTrader 5 instalado y configurado
- Cuenta de trading (demo o real) con acceso a EUR/USD

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

# Bot de Trading EUR/USD

Sistema de predicción para el par EUR/USD utilizando aprendizaje automático. Este bot obtiene datos desde MetaTrader 5, aplica indicadores técnicos, y predice la dirección del mercado (subida, bajada o neutral).

## Instalación

1. Clonar este repositorio
2. Instalar las dependencias:

```bash
cd eurusd_bot
invoke install_dependencies
```

Nota: La instalación de TA-Lib puede ser complicada en algunos sistemas. Si tiene problemas, consulte las [instrucciones específicas de instalación de TA-Lib](https://github.com/mrjbq7/ta-lib).

## Configuración

1. Asegúrese que MetaTrader 5 esté instalado y en ejecución
2. Verifique que tenga acceso al par EUR/USD en su cuenta

## Uso del Sistema

### Preparación del modelo

Para entrenar el modelo desde cero:

```bash
invoke all
```

Esto ejecutará todo el pipeline: limpieza, preprocesamiento, división de datos y entrenamiento.

### Obtención de datos en tiempo real

Para obtener datos recientes de MetaTrader:

```bash
invoke fetch_live
```

### Predicciones

Para realizar una predicción única:

```bash
invoke predict
```

Para ejecutar predicciones continuas (cada 30 segundos):

```bash
invoke predict --continuous
```

### Modo producción

Para ejecutar el sistema en modo producción:

```bash
invoke production
```

## Estructura del sistema

- `src/`: Código fuente
  - `fetch_metatrader_data.py`: Obtiene datos de MetaTrader 5
  - `preprocessing.py`: Preprocesamiento de datos históricos
  - `train_model.py`: Entrena el modelo XGBoost
  - `predict.py`: Realiza predicciones usando el modelo entrenado
  - `split_data.py`: Divide los datos en conjuntos de entrenamiento y prueba
- `data/`: Datos del sistema
  - `processed/`: Datos procesados
  - `model/`: Modelo entrenado y datos para entrenamiento
  - `market/`: Datos en tiempo real y predicciones
- `tasks.py`: Tareas automatizadas (usando invoke)

## Notas importantes

1. El sistema está configurado por defecto para usar simulación. Para usar datos reales de MetaTrader, cambie `usar_simulacion = True` a `False` en el archivo `src/predict.py`.

2. Para un rendimiento óptimo, asegúrese de tener MetaTrader 5 abierto y conectado a su broker mientras usa el sistema.

3. Este sistema es solo con fines educativos. No se recomienda su uso para trading real sin pruebas exhaustivas.

## Personalización

- Puede modificar los indicadores técnicos en `src/fetch_metatrader_data.py`
- Los parámetros del modelo se pueden ajustar en `src/train_model.py`
- Para cambiar el intervalo de tiempo, modifique el parámetro `timeframe` en `src/fetch_metatrader_data.py`

