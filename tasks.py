import os
import shutil
import logging
from invoke import task

# Configuración básica del logger
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def run_script(ctx, script_path):
    logging.info(f"Ejecutando {script_path}...")
    ctx.run(f"python {script_path}")

@task
def clean(ctx):
    """Elimina archivos generados de modelos previos."""
    folder = "data/model"
    removed_files = []
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            if filename.endswith(".csv") or filename.endswith(".json") or filename.endswith(".pkl"):
                filepath = os.path.join(folder, filename)
                os.remove(filepath)
                removed_files.append(filepath)
    if removed_files:
        logging.info(f"Archivos eliminados: {removed_files}")
    else:
        logging.info("No hay archivos para eliminar.")

@task
def preprocess(ctx):
    """Preprocesa los datos crudos y genera el archivo con indicadores y target."""
    run_script(ctx, "src/preprocessing.py")

@task
def split(ctx):
    """Divide el dataset procesado en conjuntos de entrenamiento y prueba."""
    run_script(ctx, "src/split_data.py")

@task
def train(ctx):
    """Entrena el modelo con los datos procesados y lo guarda."""
    run_script(ctx, "src/train_model.py")

@task
def predict(ctx, continuous=False):
    """Ejecuta el sistema de predicción."""
    if continuous:
        logging.info("Iniciando predicciones continuas...")
        ctx.run("python -c \"import sys; sys.path.append('src'); from predict import ejecutar_prediccion_continua; ejecutar_prediccion_continua('data/model/modelo_clasificacion.pkl', intervalo_segundos=30)\"")
    else:
        logging.info("Ejecutando predicción única...")
        run_script(ctx, "src/predict.py")

@task
def fetch_live(ctx):
    """Obtiene datos en tiempo real de MetaTrader y los guarda."""
    logging.info("Obteniendo datos en tiempo real de MetaTrader...")
    run_script(ctx, "src/fetch_metatrader_data.py")

@task(pre=[clean, preprocess, split, train])
def all(ctx):
    """Ejecuta todo el pipeline completo: limpieza, preprocesamiento, split y entrenamiento."""
    logging.info("Pipeline completo ejecutado correctamente.")

@task(pre=[train])
def production(ctx):
    """Prepara y ejecuta el sistema para predicciones en producción."""
    logging.info("Iniciando sistema de predicción en modo producción...")
    run_script(ctx, "src/predict.py")
    logging.info("Para predicciones continuas, use 'invoke predict --continuous'")

@task
def install_dependencies(ctx):
    """Instala las dependencias necesarias para el proyecto."""
    logging.info("Instalando dependencias...")
    ctx.run("pip install pandas numpy scikit-learn joblib xgboost MetaTrader5 ta")
    logging.info("Dependencias instaladas correctamente.")
