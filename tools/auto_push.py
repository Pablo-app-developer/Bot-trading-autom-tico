import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from datetime import datetime

class ChangeHandler(FileSystemEventHandler):
    def on_any_event(self, event):
        if event.is_directory:
            return

        # Verifica cambios reales en archivos rastreados por Git
        result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        lines = result.stdout.strip().split("\n")
        tracked_changes = [line for line in lines if line and not line.startswith("??")]

        if not tracked_changes:
            print("🟢 Solo archivos nuevos o temporales. No se sube nada.")
            return

        print("📦 Cambios detectados, subiendo a GitHub...")

        subprocess.run(["git", "add", "."])
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        subprocess.run(["git", "commit", "-m", f"🔄 Auto commit: {timestamp}"])
        subprocess.run(["git", "push", "origin", "main"])

if __name__ == "__main__":
    path = "."  # Observa toda la carpeta actual
    event_handler = ChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path=path, recursive=True)
    observer.start()

    print("🚀 Monitoreo iniciado. Presiona Ctrl+C para detener.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
