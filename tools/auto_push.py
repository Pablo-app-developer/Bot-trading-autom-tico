import subprocess
import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

EXCLUDE_EXTENSIONS = ['.pyc', '.log']
EXCLUDE_FOLDERS = ['__pycache__', '.git']
EXCLUDE_FILES = ['.env', '.DS_Store']

class AutoPushHandler(FileSystemEventHandler):
    def on_any_event(self, event):
        if event.is_directory or any(x in event.src_path for x in EXCLUDE_FOLDERS):
            return
        if any(event.src_path.endswith(ext) for ext in EXCLUDE_EXTENSIONS):
            return
        if any(os.path.basename(event.src_path) == file for file in EXCLUDE_FILES):
            return

        print(f"[AUTO PUSH] Detectado cambio en: {event.src_path}")
        try:
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(["git", "commit", "-m", "🚀 Auto update from local changes"], check=True)
            subprocess.run(["git", "push"], check=True)
            print("✅ Cambios subidos automáticamente al repositorio.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error al ejecutar git: {e}")

if __name__ == "__main__":
    path = "."  # raíz del proyecto
    print(f"🟡 Observando cambios en: {os.path.abspath(path)}")
    event_handler = AutoPushHandler()
    observer = Observer()
    observer.schedule(event_handler, path=path, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(2)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
