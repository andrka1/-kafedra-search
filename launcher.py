"""
Точка входа для PyInstaller.
При запуске exe:
  1. Определяет рабочую папку (рядом с exe).
  2. Если нет vecdb\ - запускает индексацию.
  3. Поднимает FastAPI через uvicorn на 127.0.0.1:8000.
  4. Открывает браузер.
"""
import os
import sys
import time
import threading
import webbrowser

# --- 1. Определяем базовую папку ---
if getattr(sys, "frozen", False):
    # Запущено из собранного PyInstaller exe.
    # Работаем рядом с exe, а не в распакованном _MEIPASS.
    APP_DIR = os.path.dirname(sys.executable)
    BUNDLE_DIR = sys._MEIPASS  # type: ignore[attr-defined]
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    BUNDLE_DIR = APP_DIR

os.chdir(APP_DIR)

# Чтобы sentence-transformers и chromadb кешировали модель/базу в папке рядом с exe:
os.environ.setdefault("HF_HOME", os.path.join(APP_DIR, ".cache", "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(APP_DIR, ".cache", "huggingface"))

# --- 2. Если базы нет - запускаем индексацию ---
vecdb_path = os.path.join(APP_DIR, "vecdb")
if not os.path.isdir(vecdb_path) or not any(os.scandir(vecdb_path)):
    print("=" * 60)
    print("  Первый запуск: индексирую учебные материалы")
    print("  Это займёт 30-90 секунд, потом сервер запустится")
    print("=" * 60)
    try:
        # Предпочитаем вызывать как модуль, если есть main().
        import ingest  # noqa
        if hasattr(ingest, "main"):
            ingest.main()
        else:
            # Fallback: выполняем как скрипт.
            import runpy
            runpy.run_module("ingest", run_name="__main__")
    except Exception as e:
        print(f"[!] Ошибка индексации: {e}")
        print("    Проверьте, что в папке docs\\ есть файлы.")
        input("Enter чтобы выйти...")
        sys.exit(1)

# --- 3. Открываем браузер через 2 секунды ---
def _open_browser():
    time.sleep(2)
    webbrowser.open("http://127.0.0.1:8000")

threading.Thread(target=_open_browser, daemon=True).start()

# --- 4. Запускаем FastAPI ---
print()
print("=" * 60)
print("  Сервер запущен: http://127.0.0.1:8000")
print("  Чтобы остановить - закройте это окно.")
print("=" * 60)
print()

import uvicorn
from web.app import app

uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
