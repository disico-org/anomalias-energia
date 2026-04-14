"""
Entry point para DISICO Dashboard en EasyPanel.
Este archivo es el punto de entrada para Gunicorn.
"""
import os
import sys
import traceback
from pathlib import Path

# Añadir directorio src al path para imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Verificar archivos de datos antes de importar
data_path = Path(os.environ.get("DATA_PATH", Path(__file__).parent / "data"))
print(f"[app] DATA_PATH={data_path}", flush=True)
print(f"[app] Archivos en DATA_PATH:", flush=True)
if data_path.exists():
    for f in sorted(data_path.iterdir()):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name} ({size_mb:.2f} MB)", flush=True)
else:
    print(f"  ⚠ La carpeta {data_path} no existe", flush=True)

try:
    print("[app] Importando dashboard...", flush=True)
    from Dash import app
    server = app.server
except Exception:
    print("[app] ⛔ Error al importar Dash:", flush=True)
    traceback.print_exc()
    sys.exit(1)

if __name__ == "__main__":
    print("[app] Iniciando servidor...", flush=True)
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8050)),
        debug=False
    )