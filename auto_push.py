import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class GitAutoPusher(FileSystemEventHandler):
    def on_any_event(self, event):
        if event.is_directory or event.event_type == 'modified':
            print("📦 Cambios detectados, subiendo a GitHub...")
            subprocess.run(["git", "add", "."])
            subprocess.run(["git", "commit", "-m", "🔄 Auto: cambios detectados"])
            subprocess.run(["git", "push", "origin", "main"])  # Ajusta si usas otra rama

if __name__ == "__main__":
    path = "."  # directorio a observar
    event_handler = GitAutoPusher()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    print("👀 Observando cambios... (Ctrl+C para detener)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
