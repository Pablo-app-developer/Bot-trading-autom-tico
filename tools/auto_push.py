import os
import subprocess
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def hay_cambios():
    result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
    return bool(result.stdout.strip())

def hacer_push():
    if hay_cambios():
        logging.info("📦 Cambios detectados, subiendo a GitHub...")

        subprocess.run(['git', 'add', '.'])
        nombre_commit = f"Actualización automática desde {os.path.basename(os.getcwd())}"
        subprocess.run(['git', 'commit', '-m', nombre_commit])
        subprocess.run(['git', 'push', 'origin', 'main'])
        
        logging.info("✅ Cambios subidos correctamente.")
    else:
        logging.info("🔍 Sin cambios, no se sube nada.")

if __name__ == "__main__":
    while True:
        hacer_push()
        time.sleep(15)  # Verifica cada 15 segundos
