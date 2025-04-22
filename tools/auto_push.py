import subprocess
import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

EXCLUDE_EXTENSIONS = ['.pyc', '.log']
EXCLUDE_FOLDERS = ['__pycache__', '.git']
EXCLUDE_FILES = ['.env', '.DS_Store', 'auto_push.py']  # ✅ Evita autosubidas infinitas

class AutoPushHandler(FileSystemEventHandler):
    def on_any_event(self, event):
        if event.is_directory or any(folder in event.src_path for folder in EXCLUDE_FOLDERS):
            return
        if any(event.src_path.endswith(ext) for ext in EXCLUDE_EXTENSIONS):
            return
        if os.path.basename(event.src_path) in EXCLUDE_FILES:
            return

        print(f"📦 Cambio detectado en: {event.src_path}")
        try:
            result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
            if result.stdout.strip() == "":
                print("🟢 No hay cambios reales para subir.")
                return

            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(["git", "commit", "-m", "🚀 Auto update from local changes"], check=True)
            subprocess.run(["git", "push"], check=True)
            print("✅ Cambios subidos automáticamente a GitHub.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error ejecutando git: {e}")

if __name__ == "__main__":
    path = "."
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
